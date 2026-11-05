#!/usr/bin/env python
# =============================================================================
# OmniEchoBench-QA scorer (LLM extraction + deterministic rule-based judging).
# Evaluates the indexed prediction file supplied through --predictions.
#
# How judging works:
#   - Gold answers come from OmniEchoBench-QA.jsonl (meta + reference), matched
#     by line index.
#   - Cognitive-map multiple-choice items: these samples have empty meta
#     (category=None) and ask to "Answer with a single letter (A, B, C, or D)".
#     The gold answer is one letter; the prediction is scored by extracting the
#     chosen letter and comparing it exactly. The LLM only pulls the final letter
#     out of free text (a bare letter takes a local regex fast path).
#   - Direction (strict two-axis set equality), angle (magnitude within tol +
#     direction), and distance (within tol + direction) are all rule-based.
#
# Usage (the extraction step uses any OpenAI-compatible endpoint; the API key is read
# from OPENAI_API_KEY):
#   python qa_eval/eval_qa_metrics.py --gold /path/to/gold.jsonl \
#       --predictions /path/to/predictions.jsonl --output-dir /path/to/results
#   Offline re-judge (change tol/dir_mode without calling the LLM again):
#   python qa_eval/eval_qa_metrics.py --gold /path/to/gold.jsonl \
#       --predictions /path/to/predictions.jsonl --output-dir /path/to/results \
#       --rejudge --rewrite
# =============================================================================
import argparse
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.environ.get("OMNIECHO_BENCH_ROOT", os.path.join(os.path.dirname(HERE), "data", "OmniEchoBench"))
VAL = os.environ.get("OMNIECHO_QA_GOLD", os.path.join(BENCH_ROOT, "OmniEchoBench-QA", "OmniEchoBench-QA.jsonl"))
# Directory for prediction files and judgment details (defaults to qa_eval/output/);

OUTDIR = os.environ.get("OMNIECHO_QA_OUTPUT", os.path.join(HERE, "output"))
# The extraction step uses any OpenAI-compatible endpoint. Override with --base_url.
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.example.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "OmniEcho")

# Internal key for the cognitive-map multiple-choice category (these samples have
# empty meta in the test set).
TOPDOWN_CAT = "topdown"

# Display names used in reports; internal keys are kept for data matching.
CATEGORY_DISPLAY = {
    "camera_rotation": "camera rotation",
    "motion": "source motion",
    "direction": "source direction",
    "localization3d": "3D localization",
    "topdown": "cognitive map",
}


def display_cat(cat):
    """Map an internal category key to its report display name."""
    return CATEGORY_DISPLAY.get(cat, cat)


INHERIT_TOPDOWN = {}

# Files to evaluate (short name -> filename under output/). Missing files are skipped.
PRED_FILES = {"OmniEcho": "predictions.jsonl"}

MODEL_NAMES = {"OmniEcho": "OmniEcho"}

# ---------------------------------------------------------------------------
# Direction / numeric parsing (shared vocabulary for gold and rule-based judging).
# ---------------------------------------------------------------------------
DIR_TOKENS = {
    "left": re.compile(r"\bleft\b", re.I),
    "right": re.compile(r"\bright\b", re.I),
    "front": re.compile(r"\b(front|ahead|forward|in front)\b", re.I),
    "back": re.compile(r"\b(back|behind|rear|backward)\b", re.I),
}
OPPOSITE = {"left": "right", "right": "left", "front": "back", "back": "front"}


def parse_dir_set(text):
    """Parse the set of directions (subset of left/right/front/back) from text."""
    text = text or ""
    return {k for k, rx in DIR_TOKENS.items() if rx.search(text)}


def first_number(text):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m.group()) if m else None


def distance_meters(text):
    """Return the last number expressed like '2.5 m' / '2.5 meters'."""
    ms = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*m(?:eters?|etres?)?\b", str(text), re.I)
    return float(ms[-1]) if ms else None


def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Cognitive-map MC: extract a single option letter (A/B/C/D) from gold / prediction.
# ---------------------------------------------------------------------------
LETTER_RX = re.compile(r"\b([ABCD])\b")


def gold_letter(text):
    """Gold is usually a single letter; fall back to the first A-D in the text."""
    t = str(text or "").strip().upper()
    if t in ("A", "B", "C", "D"):
        return t
    m = LETTER_RX.search(t)
    return m.group(1) if m else None


def rule_letter(pred):
    """Local fast path: return a letter only when the prediction clearly points to
       one, else None (let the LLM extract it). It accepts a whole-string letter
       or the last recognized 'option/answer/location/choice X' phrase. Other
       wording, including 'the answer is X', may require LLM extraction."""
    t = (pred or "").strip()
    if t.upper() in ("A", "B", "C", "D"):
        return t.upper()
    # "the answer is D" / "option (D)" / "choice: B" / "location A" ...
    phrase = re.findall(
        r"(?:answer|option|choice|choose|chose|location|marked|letter|select(?:ed)?|pick(?:ed)?)"
        r"[^A-Za-z0-9]{0,6}\(?\b([ABCD])\b\)?",
        t, re.I)
    if phrase:
        return phrase[-1].upper()
    return None  # unsure -> let the LLM extract the letter


def log(msg):
    print(f"[eval] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Global rate limiter: most OpenAI-compatible endpoints throttle per key by RPM
# and return 429 when exceeded, and retries themselves keep burning quota — so the
# total request rate must be capped client-side rather than pushed through with more
# workers/retries. This releases requests at a fixed interval (no bursting) to suit
# sliding-window counters.
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rpm):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._next <= now:
                self._next = now + self.min_interval
                wait = 0.0
            else:
                wait = self._next - now
                self._next += self.min_interval
        if wait > 0:
            time.sleep(wait)


RATE_LIMITER = None   # initialized in main() from --rpm


# ---------------------------------------------------------------------------
# LLM extraction: only normalizes the MODEL prediction into a structure; it does
# not decide correctness.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You normalize a model's free-text answer for a spatial-audio video QA benchmark.
You are given the QUESTION and the MODEL_ANSWER (may contain reasoning; the real
answer is usually its last line). You also get the GOLD_ANSWER only so you can give
a fallback is_correct; do NOT let it change how you extract the model's own answer.

Extract, FROM THE MODEL_ANSWER ONLY, these normalized fields:
  - yesno: "yes" or "no" if the model answer is a yes/no answer, else null.
  - dir_set: the list of horizontal directions the model asserts about the target
      (the sound source location, its motion, or the camera's rotation direction),
      each one of "left","right","front","back". Map ahead/forward->front,
      behind/rear/backward->back. Include EVERY direction the answer commits to
      (e.g. "behind and to the left" -> ["back","left"]); use [] if none/ambiguous.
      Do NOT include a direction the answer negates (e.g. "not to the left").
  - degrees: the magnitude of rotation/turn in degrees as a float if the answer
      states an angle, else null.
  - distance_m: the distance to the source in METERS as a float if stated, else null.
  - letter: for a multiple-choice question that asks to answer with a single letter
      (A, B, C or D), the letter the model finally chooses, one of "A","B","C","D";
      else null. Pick the model's FINAL choice, not options it merely discusses.
  - answer: the model's final short answer as a single normalized string.

Then give a FALLBACK judgment (used only when the program cannot decide by rule):
  - is_correct: boolean, whether the model answer means the same as GOLD_ANSWER
      (synonyms/paraphrase ok; opposite direction, wrong number, or wrong letter is wrong).
  - reason: one short sentence.

Return ONLY a JSON object, no markdown fences, exactly these keys:
{"yesno": "yes"|"no"|null, "dir_set": [...], "degrees": <float|null>,
 "distance_m": <float|null>, "letter": "A"|"B"|"C"|"D"|null,
 "answer": "...", "is_correct": true|false, "reason": "..."}
"""


def parse_json_reply(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    l, r = t.find("{"), t.rfind("}")
    if l != -1 and r != -1 and r > l:
        t = t[l:r + 1]
    return json.loads(t)


def strip_think(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def last_line(s):
    if not isinstance(s, str):
        return ""
    for ln in reversed(s.splitlines()):
        if ln.strip():
            return ln.strip()
    return ""


def build_user_content(question, prediction, gold):
    return (
        f"QUESTION:\n{question.strip()}\n\n"
        f"MODEL_ANSWER (final answer is usually the last line):\n{prediction.strip()}\n\n"
        f"Extract and return only the JSON object. There is no GOLD_ANSWER; set "
        f"is_correct to false and reason to \"no-gold\"."
    )


# ---------------------------------------------------------------------------
# Deterministic rule-based judging (gold from val meta + reference).
# ---------------------------------------------------------------------------
def dir_set_match(gold, pred, mode):
    gold, pred = set(gold), set(pred)
    if mode == "strict":
        return gold == pred
    if not gold.issubset(pred):
        return False
    return not any(OPPOSITE[g] in pred for g in gold)


def turn_dir_ok(gold_dir, pred_dir):
    gold_dir, pred_dir = set(gold_dir), set(pred_dir)
    axis = (gold_dir & {"left", "right"}) or (gold_dir & {"front", "back"})
    if not axis:
        return True
    for g in axis:
        if g not in pred_dir or OPPOSITE[g] in pred_dir:
            return False
    return True


def judge_by_rule(gt, ex, angle_tol, dist_tol, dir_mode):
    """Return (is_correct: 0/1, or None if undecidable; detail: str)."""
    cat, sub, bucket, ref = gt["category"], gt["subtype"], gt["bucket"], gt["ref"]

    # --- cognitive-map MC (deterministic): extracted letter must match exactly ---
    if cat == TOPDOWN_CAT:
        gl = gold_letter(ref)
        pl = ex.get("letter")
        pl = str(pl).strip().upper() if pl else None
        if pl not in ("A", "B", "C", "D"):
            return None, f"letter-unparsed gold={gl}"
        if gl is None:
            return None, "gold-letter-missing"
        return int(pl == gl), f"letter pred={pl} gold={gl}"

    # --- yes/no ---
    if bucket in ("yes", "no"):
        py = ex.get("yesno")
        if py in ("yes", "no"):
            return int(py == bucket), f"yesno pred={py} gold={bucket}"
        return None, "yesno-unparsed"

    # --- angle ---
    if (cat == "camera_rotation" and sub == "degrees") or \
       (cat == "direction" and sub == "degrees_body"):
        gdeg = first_number(ref)
        pdeg = to_float(ex.get("degrees"))
        pdir = set(ex.get("dir_set") or [])
        if gdeg is None:
            return None, "gold-deg-missing"
        if pdeg is None:
            return 0, "pred-deg-missing"
        dir_ok = turn_dir_ok(parse_dir_set(ref), pdir) if dir_mode == "strict" else (bucket in pdir)
        ok = (abs(pdeg - gdeg) <= angle_tol) and dir_ok
        return int(ok), f"deg |{pdeg}-{gdeg}|<= {angle_tol}?{abs(pdeg-gdeg)<=angle_tol} dir_ok={dir_ok}"

    # --- distance ---
    if cat == "localization3d":
        gdist = distance_meters(ref)
        pdist = to_float(ex.get("distance_m"))
        pdir = set(ex.get("dir_set") or [])
        if gdist is None:
            return None, "gold-dist-missing"
        if pdist is None:
            return 0, "pred-dist-missing"
        dir_ok = dir_set_match(parse_dir_set(ref), pdir, "strict") if dir_mode == "strict" \
            else (bucket in pdir)
        ok = (abs(pdist - gdist) <= dist_tol) and dir_ok
        return int(ok), f"dist |{pdist}-{gdist}|<= {dist_tol}?{abs(pdist-gdist)<=dist_tol} dir_ok={dir_ok}"

    # --- pure direction ---
    if bucket in DIR_TOKENS:
        pdir = set(ex.get("dir_set") or [])
        if dir_mode == "strict":
            gdir = parse_dir_set(ref)
            if not gdir:
                return None, "gold-dir-missing"
            return int(gdir == pdir), f"dir[strict] {sorted(gdir)}vs{sorted(pdir)}"
        return int(bucket in pdir), f"dir[bucket] {bucket}∈{sorted(pdir)}"

    return None, "no-rule"


def judge_one(client, model, item, angle_tol, dist_tol, dir_mode, retries, timeout):
    gt = item["gt"]
    question = item["question"]
    prediction = item["prediction"]
    gold = last_line(gt["ref"]) or gt["ref"]

    # cognitive-map MC: if the local regex already fixes the letter, judge directly
    # and save an LLM call.
    if gt["category"] == TOPDOWN_CAT:
        pl = rule_letter(prediction)
        if pl is not None:
            gl = gold_letter(gt["ref"])
            return {
                "index": item["index"], "category": TOPDOWN_CAT, "subtype": gt["subtype"],
                "bucket": gt["bucket"],
                "extracted": {"letter": pl, "dir_set": [], "yesno": None,
                              "degrees": None, "distance_m": None, "answer": pl},
                "llm_correct": bool(gl is not None and pl == gl),
                "is_correct": bool(gl is not None and pl == gl),
                "verdict_source": "rule", "detail": f"letter[local] pred={pl} gold={gl}",
                "reason": "local letter regex", "pred_last": last_line(prediction), "gold": gold,
            }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(question, prediction, gold)},
    ]
    last_err = None
    for attempt in range(retries):
        try:
            if RATE_LIMITER is not None:
                RATE_LIMITER.acquire()     # retries obey the rate limit too, else backoff bursts hit the RPM cap again
            completion = client.chat.completions.create(
                model=model, messages=messages, timeout=timeout, temperature=0,
            )
            data = parse_json_reply(completion.choices[0].message.content)
            ex = {
                "yesno": data.get("yesno"),
                "dir_set": data.get("dir_set") or [],
                "degrees": data.get("degrees"),
                "distance_m": data.get("distance_m"),
                "letter": data.get("letter"),
                "answer": data.get("answer"),
            }
            llm_correct = bool(data.get("is_correct", False))
            rule_correct, detail = judge_by_rule(gt, ex, angle_tol, dist_tol, dir_mode)
            if rule_correct is None:
                final, source = 0, "llm-blind-wrong"
            else:
                final, source = int(rule_correct), "rule"
            return {
                "index": item["index"],
                "category": gt["category"],
                "subtype": gt["subtype"],
                "bucket": gt["bucket"],
                "extracted": ex,
                "llm_correct": llm_correct,
                "is_correct": bool(final),
                "verdict_source": source,
                "detail": detail,
                "reason": str(data.get("reason", ""))[:300],
                "pred_last": last_line(prediction),
                "gold": gold,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries - 1:
                # Upstream quota jitter intermittently returns 429 (insufficient_quota /
                # non_2xx); a retry usually gets through -> exponential backoff + jitter
                # (plain linear backoff cannot get through under high concurrency).
                time.sleep(min(2 ** attempt, 20) + random.uniform(0, 1.5))
    return {"index": item["index"], "category": gt["category"], "subtype": gt["subtype"],
            "is_correct": False, "verdict_source": "ERROR", "error": str(last_err)}


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------
def load_gt():
    gt = {}
    with open(VAL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m = d.get("meta") or {}
            ref = d["messages"][-1]["content"]
            if isinstance(ref, list):
                ref = " ".join(c.get("text", "") for c in ref if isinstance(c, dict))
            ref = str(ref)
            cat = m.get("category")
            if cat == "visibility":               # visibility metric removed: not evaluated
                continue
            if cat is None:                        # empty meta => cognitive-map MC
                gt[i] = {"category": TOPDOWN_CAT, "subtype": "mc",
                         "bucket": gold_letter(ref), "ref": ref}
            else:
                gt[i] = {"category": cat, "subtype": m.get("subtype"),
                         "bucket": m.get("answer_bucket"), "ref": ref}
    return gt


def extract_question(rec):
    q = rec.get("question")
    if q:
        return q
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        texts = []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text", ""))
        if texts:
            return "\n".join(texts)
    return ""


def load_pred_file(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "index" in r:
                try:
                    recs[int(r["index"])] = r   # normalize to int to align with gold enumerate
                except (TypeError, ValueError):
                    recs[r["index"]] = r
    return recs


def load_done(out_path):
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("verdict_source") != "ERROR":
                        done[r["index"]] = r
                except Exception:  # noqa: BLE001
                    continue
    return done


def prune_judgments(out_path, drop):
    """Remove invalidated judgments before computing replacement verdicts."""
    drop = set(drop)
    if not drop:
        return 0
    kept, removed = [], 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except Exception:  # noqa: BLE001
                kept.append(s)
                continue
            if r.get("index") in drop and r.get("verdict_source") != "ERROR":
                removed += 1
                continue
            kept.append(s)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in kept:
            f.write(s + "\n")
    return removed


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _overall(by_cat):
    """Compute micro/macro from by_cat={cat:[correct,total]}."""
    items = [(c, cc, nn) for c, (cc, nn) in by_cat.items()]
    tot_c = sum(cc for _, cc, _ in items)
    tot_n = sum(nn for _, _, nn in items)
    micro = tot_c / tot_n if tot_n else 0.0
    macro = (sum(cc / nn for _, cc, nn in items if nn) / len(items)) if items else 0.0
    return micro, macro, tot_c, tot_n


def recompute_summary(s):
    """Recompute micro/macro/correct/total from s['by_cat'] (in place)."""
    s["micro"], s["macro"], s["correct"], s["total"] = _overall(s["by_cat"])
    return s


def apply_topdown_inheritance(summaries):
    """Copy the source model's cognitive-map score onto the target model (when the
       target has none), then recompute the target's overall. Returns the list of
       changed target keys (for printing/annotation)."""
    changed = []
    for tgt, src in INHERIT_TOPDOWN.items():
        if tgt not in summaries or src not in summaries:
            continue
        tgt_top = summaries[tgt]["by_cat"].get(TOPDOWN_CAT, (0, 0))
        src_top = summaries[src]["by_cat"].get(TOPDOWN_CAT, (0, 0))
        if tgt_top[1] > 0:          # target already has cognitive-map items, do not override
            continue
        if src_top[1] <= 0:         # source has none either, nothing to inherit
            log(f"[inherit] {tgt} wants to inherit {src}'s cognitive-map score, but {src} has none yet (skipped)")
            continue
        summaries[tgt]["by_cat"][TOPDOWN_CAT] = tuple(src_top)
        summaries[tgt]["topdown_inherited_from"] = src
        recompute_summary(summaries[tgt])
        cc, nn = src_top
        log(f"[inherit] {tgt}'s cognitive-map score inherited from {src}: {cc}/{nn}={cc/nn:.3f}; recomputed {tgt} overall")
        changed.append(tgt)
    return changed


def report(judgments, gt, title=""):
    by_cat = defaultdict(lambda: [0, 0])   # category -> [correct, total]
    src = defaultdict(int)
    err = 0
    for j in judgments:
        if j.get("verdict_source") == "ERROR":
            err += 1
            continue
        c = j.get("category") or (gt.get(j["index"], {}) or {}).get("category", "other")
        by_cat[c][1] += 1
        by_cat[c][0] += int(bool(j.get("is_correct")))
        src[j.get("verdict_source", "?")] += 1

    micro, macro, tot_c, tot_n = _overall(by_cat)

    log("=" * 68)
    if title:
        log(f"Report: {title}")
        log("-" * 68)
    log(f"{'category':<20}{'correct':>9}{'total':>8}{'acc':>9}")
    log("-" * 68)
    for c in sorted(by_cat, key=lambda k: -by_cat[k][1]):
        cc, nn = by_cat[c]
        log(f"{display_cat(c):<20}{cc:>9}{nn:>8}{(cc/nn if nn else 0):>9.3f}")
    log("-" * 68)
    log(f"{'OVERALL (micro)':<28}{tot_c:>7}{tot_n:>8}{micro:>9.3f}")
    log(f"{'OVERALL (macro)':<28}{'':>7}{'':>8}{macro:>9.3f}")
    log(f"judged by: {dict(src)}" + (f"  | failed (excluded): {err}" if err else ""))
    log("=" * 68)
    return {"micro": micro, "macro": macro, "correct": tot_c, "total": tot_n,
            "by_cat": {c: (cc, nn) for c, (cc, nn) in by_cat.items()},
            "src": dict(src), "err": err}


def eval_one_file(key, args, client, gt):
    pred_path = os.path.join(OUTDIR, PRED_FILES[key])
    if not os.path.exists(pred_path):
        log(f"[{key}] skip: output/{PRED_FILES[key]} not found")
        return None
    out_path = os.path.join(OUTDIR, f"eval_{key}_judgments.jsonl")

    raw = load_pred_file(pred_path)
    items = []
    for idx, g in gt.items():
        rec = raw.get(idx)
        if rec is None:
            continue
        pred = strip_think(rec.get("prediction", "") or "")
        if not pred.strip():
            continue
        q = extract_question(rec) or ""
        items.append({"index": idx, "gt": g, "question": q, "prediction": pred})
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    log(f"[{key}] samples to judge: {len(items)} (from output/{PRED_FILES[key]}, {len(raw)} raw records)")

    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    done = load_done(out_path)
    if done:
        log(f"[{key}] resume: {len(done)} already judged, skipping.")
    # The prediction file may be regenerated after judging (e.g. a splice script
    # rewrote the cognitive-map segment), while resume only dedups by index — so
    # results judged against the old prediction would be silently reused. Compare
    # pred_last per record and drop any that no longer match, forcing a re-judge.
    pred_now = {it["index"]: last_line(it["prediction"]).strip() for it in items}
    stale = {i for i, j in done.items()
             if i in pred_now and (j.get("pred_last") or "").strip() != pred_now[i]}
    if stale:
        for i in stale:
            done.pop(i, None)
        n = prune_judgments(out_path, stale)
        log(f"[{key}] {len(stale)} old judgments have a pred_last that no longer matches the current "
            f"prediction; removed {n} stale judgments before re-judging.")
    todo = [it for it in items if it["index"] not in done]
    log(f"[{key}] to judge this run: {len(todo)}, workers {args.workers}")

    lock = threading.Lock()
    total = len(todo)
    with open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(judge_one, client, args.model, it, args.angle_tol,
                             args.dist_tol, args.dir_mode, args.retries, args.timeout)
                   for it in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            j = fut.result()
            with lock:
                fout.write(json.dumps(j, ensure_ascii=False) + "\n")
                fout.flush()
                if i <= 5 or i % 100 == 0:
                    log(f"[{key}][{i}/{total}] idx={j.get('index')} cat={j.get('category')} "
                        f"correct={j.get('is_correct')} src={j.get('verdict_source')}")

    all_j = list(load_done(out_path).values())
    log(f"[{key}] judging done, {len(all_j)} detail records -> {out_path}")
    return report(all_j, gt, title=f"{MODEL_NAMES.get(key, key)}  ({PRED_FILES[key]})")


def rejudge_one_file(key, args, gt):
    """Offline re-judge: reuse the LLM extractions in eval_<key>_judgments.jsonl,
       without calling the LLM."""
    out_path = os.path.join(OUTDIR, f"eval_{key}_judgments.jsonl")
    if not os.path.exists(out_path):
        log(f"[{key}] skip re-judge: judgment file {os.path.basename(out_path)} not found "
            f"(judge online once first)")
        return None

    recs = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue

    changed = 0
    for r in recs:
        if r.get("verdict_source") == "ERROR":
            continue
        idx = r.get("index")
        g = gt.get(idx)
        ex = r.get("extracted")
        if g is None or ex is None:
            continue
        rule_correct, detail = judge_by_rule(g, ex, args.angle_tol, args.dist_tol, args.dir_mode)
        if rule_correct is None:
            final, source = int(bool(r.get("llm_correct", False))), "llm"
        else:
            final, source = int(rule_correct), "rule"
        if bool(final) != bool(r.get("is_correct")) or source != r.get("verdict_source"):
            changed += 1
        r["is_correct"], r["verdict_source"], r["detail"] = bool(final), source, detail

    if args.rewrite:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"[{key}] wrote back updated judgments -> {os.path.basename(out_path)} ({changed} verdicts changed)")
    else:
        log(f"[{key}] offline re-judge done (not written; {changed} verdicts changed; add --rewrite to persist)")

    return report(recs, gt, title=f"{MODEL_NAMES.get(key, key)}  ({PRED_FILES.get(key, key)}) [rejudge]")


def print_comparison(summaries, gt, args):
    keys = list(summaries.keys())
    cats = sorted({v["category"] for v in gt.values()})
    header = f"{'category':<18}" + "".join(f"{k:>10}" for k in keys)
    log("")
    log("#" * len(header))
    log("Cross-model comparison (accuracy) — direction strict two-axis, angle +-%.1f deg, "
        "distance +-%.1fm, cognitive-map option letter exact match" %
        (args.angle_tol, args.dist_tol))
    log("#" * len(header))
    log(header)
    log("-" * len(header))
    for c in cats:
        row = f"{display_cat(c):<18}"
        for k in keys:
            cc, nn = summaries[k]["by_cat"].get(c, (0, 0))
            row += (f"{(cc/nn):>10.3f}" if nn else f"{'-':>10}")
        log(row)
    log("-" * len(header))
    log(f"{'OVERALL(micro)':<18}" + "".join(f"{summaries[k]['micro']:>10.3f}" for k in keys))
    log(f"{'OVERALL(macro)':<18}" + "".join(f"{summaries[k]['macro']:>10.3f}" for k in keys))
    log(f"{'#judged':<18}" + "".join(f"{summaries[k]['total']:>10d}" for k in keys))
    log("#" * len(header))


def main():
    global VAL, OUTDIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=VAL, help="OmniEchoBench-QA gold JSONL.")
    ap.add_argument("--predictions", required=True, help="Indexed OmniEcho predictions JSONL.")
    ap.add_argument("--output-dir", default=OUTDIR)
    ap.add_argument("--files", nargs="+", default=list(PRED_FILES.keys()),
                    choices=list(PRED_FILES.keys()), help="short names of files to evaluate (default: all)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="text model used for judging/extraction")
    ap.add_argument("--api_key",
                    default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key (defaults to OPENAI_API_KEY)")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL,
                    help="OpenAI-compatible endpoint (default %s)" % DEFAULT_BASE_URL)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rpm", type=int, default=55,
                    help="global request rate cap (per minute); 0 = unlimited")
    ap.add_argument("--angle_tol", type=float, default=45, help="angle tolerance (degrees)")
    ap.add_argument("--dist_tol", type=float, default=2.0, help="distance tolerance (meters)")
    ap.add_argument("--dir_mode", default="strict", choices=["strict", "bucket"],
                    help="direction judging: strict=exact two-axis set equality (default); bucket=loose containment")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=8,
                    help="max attempts per record (exponential backoff on 429/jitter)")
    ap.add_argument("--limit", type=int, default=0, help=">0 evaluates only the first N records (debug)")
    ap.add_argument("--overwrite", action="store_true", help="clear existing judgments and start over")
    ap.add_argument("--rejudge", action="store_true",
                    help="offline re-judge: reuse the LLM extractions in the judgment file and recompute with "
                         "the current tol/dir_mode, without calling the LLM")
    ap.add_argument("--rewrite", action="store_true",
                    help="with --rejudge: write the recomputed verdicts back to the judgment file (default: compute only)")
    args = ap.parse_args()
    VAL = os.path.abspath(args.gold)
    OUTDIR = os.path.abspath(args.output_dir)
    PRED_FILES["OmniEcho"] = os.path.abspath(args.predictions)
    if not os.path.isfile(PRED_FILES["OmniEcho"]):
        ap.error("Prediction file does not exist.")
    os.makedirs(OUTDIR, exist_ok=True)

    gt = load_gt()
    n_top = sum(1 for g in gt.values() if g["category"] == TOPDOWN_CAT)
    log(f"gold: {len(gt)} records (test meta + reference), of which {n_top} cognitive-map MC | "
        f"dir_mode={args.dir_mode} angle_tol={args.angle_tol} deg dist_tol={args.dist_tol}m"
        + ("  [REJUDGE offline, no LLM call]" if args.rejudge else ""))

    summaries = {}
    if args.rejudge:
        for key in args.files:
            s = rejudge_one_file(key, args, gt)
            if s is not None:
                summaries[key] = s
    else:
        if not args.api_key:
            raise SystemExit("provide a key via --api_key or the OPENAI_API_KEY environment variable")
        from openai import OpenAI  # lazy import
        global RATE_LIMITER
        RATE_LIMITER = RateLimiter(args.rpm)
        log(f"endpoint={args.base_url} model={args.model} workers={args.workers} "
            f"rpm_limit={args.rpm or 'unlimited'} retries={args.retries}")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        for key in args.files:
            s = eval_one_file(key, args, client, gt)
            if s is not None:
                summaries[key] = s

    # No inheritance occurs with the current empty INHERIT_TOPDOWN mapping.
    apply_topdown_inheritance(summaries)

    if len(summaries) > 1:
        print_comparison(summaries, gt, args)

    # Write a machine-readable summary.
    out = {MODEL_NAMES.get(k, k): {
        "file": PRED_FILES[k], "judged": s["total"],
        "overall": {"micro": round(s["micro"], 4), "macro": round(s["macro"], 4),
                    "correct": s["correct"], "total": s["total"]},
        "verdict_source": s["src"],
        "topdown_inherited_from": s.get("topdown_inherited_from"),
        "per_category": {display_cat(c): {"correct": cc, "total": nn, "acc": round(cc / nn, 4) if nn else None}
                         for c, (cc, nn) in s["by_cat"].items()},
    } for k, s in summaries.items()}
    out["_config"] = {"dir_mode": args.dir_mode, "angle_tol": args.angle_tol,
                      "dist_tol": args.dist_tol, "model": args.model}
    out_json = os.path.join(OUTDIR, "eval_qa_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"summary -> {out_json}")


if __name__ == "__main__":
    main()
