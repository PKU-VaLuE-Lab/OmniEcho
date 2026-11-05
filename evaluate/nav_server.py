#!/usr/bin/env python
# =============================================================================

# -----------------------------------------------------------------------------


#     POST /infer {content_parts:[{type:text|image|audio, value}],
#                  max_new_tokens, assistant_prefix?}
#         -> {trajectory:[[x,y,yaw],...], raw, has_foa, n_foa_pad}
#     GET  /health -> {status:"ok", ...}
#


#

#

#   CUDA_VISIBLE_DEVICES=0 python evaluate/nav_server.py \
#       --model-path .../global_step_2000/hf_ckpt_infer --host 0.0.0.0 --port 8210
# =============================================================================
import argparse
import importlib.util
import json
import os
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_INFER_PATH = os.path.join(REPO_ROOT, "evaluate", "infer_qa.py")
_spec = importlib.util.spec_from_file_location("infer_qa", _INFER_PATH)
infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(infer)

OmniEchoModel = infer.OmniEchoModel
OmniEchoProcessor = infer.OmniEchoProcessor
process_mm_info = infer.process_mm_info
foa_num_channels = infer.foa_num_channels
foa_path_to_encoder_features = infer.foa_path_to_encoder_features
FOA_BOS_ID = infer.FOA_BOS_ID
FOA_PAD_ID = infer.FOA_PAD_ID
FOA_EOS_ID = infer.FOA_EOS_ID

AUDIO_TOKEN_ID = 151675  


def log(msg):
    print(f"[foa-nav-server] {msg}", flush=True)


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def _strip_cot(txt):
    if not txt:
        return txt
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL | re.IGNORECASE)
    low = txt.lower()
    if "</think>" in low:
        txt = txt[low.rfind("</think>") + len("</think>"):]
    return re.sub(r"</?think>", " ", txt, flags=re.IGNORECASE)


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_WAYPOINT_RE = re.compile(r"\[\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*\]" % (_NUM, _NUM, _NUM))


def parse_trajectory(txt):
    """OmniEcho processing component."""
    if not txt:
        return []
    body = _strip_cot(txt)
    start = body.rfind("[[")
    if start < 0:
        start = body.find("[")
    region = body[start:] if start >= 0 else body
    out = []
    for m in _WAYPOINT_RE.finditer(region):
        try:
            out.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
class OmniEchoNavModel:
    def __init__(self, model_path, dtype="bfloat16", attn="flash_attention_2",
                 use_audio_in_video=True):
        self.use_audio_in_video = use_audio_in_video
        self.dtype = getattr(torch, dtype)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        t0 = time.time()
        log(f"Loading model: {model_path} (dtype={dtype}, attn={attn})")
        self.model = OmniEchoModel.from_pretrained(
            model_path, dtype=self.dtype, device_map="auto",
            attn_implementation=attn, ignore_mismatched_sizes=True,
        )
        
        infer._fix_transposed_fused_experts(self.model, model_path, self.dtype)
        self.model.eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        log(f"Model loaded in {time.time()-t0:.1f}s")

        self.processor = OmniEchoProcessor.from_pretrained(model_path)
        vocab = self.processor.tokenizer.get_vocab()
        self.vision_end_id = vocab.get("<|vision_end|>")
        self.text_hidden = self.model.config.thinker_config.text_config.hidden_size


        emb_dev = self.model.thinker.get_input_embeddings().weight.device
        self.enc, self.proj = infer.load_foa_modules(
            model_path, emb_dev, self.dtype, self.text_hidden)
        self._stash = {"emb": None, "pos": None}

        def _prehook(module, hook_args, hook_kwargs):
            emb, pos = self._stash["emb"], self._stash["pos"]
            if emb is None or pos is None or pos.numel() == 0:
                return None
            ie = hook_kwargs.get("inputs_embeds", None)
            if ie is None or ie.dim() != 3 or ie.shape[1] <= 1:
                return None  
            ie = ie.clone()
            p = pos.to(ie.device)
            ie[0, p] = emb.to(ie.device, ie.dtype)
            hook_kwargs["inputs_embeds"] = ie
            return hook_args, hook_kwargs

        self.model.thinker.model.register_forward_pre_hook(_prehook, with_kwargs=True)
        self._lock = threading.Lock()
        log(f"Registered FOA prefill hook (text_hidden={self.text_hidden})")


    @staticmethod
    def _content_from_parts(parts):
        """OmniEcho processing component."""
        content, foa_src = [], None
        for p in parts:
            ty, val = p.get("type"), p.get("value")
            if ty == "text":
                content.append({"type": "text", "text": val})
            elif ty == "image":
                content.append({"type": "image", "image": val})
            elif ty == "audio":
                content.append({"type": "audio", "audio": val})
                if foa_src is None:
                    foa_src = val
        return content, foa_src

    def _build_inputs(self, content, foa_src, assistant_prefix=None):
        """OmniEcho processing component."""
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        if assistant_prefix:
            text = text + assistant_prefix
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video,
        )
        if foa_src is None or inputs.get("feature_attention_mask", None) is None:
            return inputs, None, None
        if foa_num_channels(foa_src) < 4:
            return inputs, None, None
        feats = foa_path_to_encoder_features(foa_src)
        if feats is None:
            return inputs, None, None

        ids = inputs["input_ids"].squeeze(0).tolist()
        attn = inputs["attention_mask"].squeeze(0).tolist()
        apad_pos = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
        if not apad_pos:
            return inputs, None, None
        L = len(apad_pos)
        insert_at = apad_pos[-1] + 1 + 1  
        if insert_at < len(ids) and ids[insert_at] == self.vision_end_id:
            insert_at += 1
        span = [FOA_BOS_ID] + [FOA_PAD_ID] * L + [FOA_EOS_ID]
        ids[insert_at:insert_at] = span
        attn[insert_at:insert_at] = [1] * len(span)
        inputs["input_ids"] = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        inputs["attention_mask"] = torch.tensor(
            attn, dtype=inputs["attention_mask"].dtype).unsqueeze(0)
        return inputs, L, feats


    @torch.no_grad()
    def infer_one(self, content_parts, max_new_tokens=512, assistant_prefix=None):
        content, foa_src = self._content_from_parts(content_parts)
        with self._lock:
            inputs, L, feats = self._build_inputs(content, foa_src, assistant_prefix)
            has_foa, n_pad = 0, 0
            try:
                if L is not None and feats is not None:
                    ids_t = inputs["input_ids"].squeeze(0)
                    pos = (ids_t == FOA_PAD_ID).nonzero(as_tuple=False).squeeze(-1)
                    n_pad = int(pos.numel())
                    if n_pad > 0:
                        emb_dev = self.model.thinker.get_input_embeddings().weight.device
                        emb = infer.compute_foa_embeds(
                            self.enc, self.proj, feats, L,
                            out_dtype=self.dtype, out_device=emb_dev)
                        self._stash["emb"], self._stash["pos"] = emb, pos
                        has_foa = 1
                    else:
                        self._stash["emb"] = self._stash["pos"] = None
                else:
                    self._stash["emb"] = self._stash["pos"] = None

                inputs = inputs.to(self.model.device).to(self.model.dtype)
                out_ids = self.model.generate(
                    **inputs, thinker_max_new_tokens=int(max_new_tokens),
                    return_audio=False, use_audio_in_video=self.use_audio_in_video,
                )
            finally:
                self._stash["emb"] = self._stash["pos"] = None  

            seq = out_ids[0] if isinstance(out_ids, (tuple, list)) else out_ids
            seq = seq.sequences if hasattr(seq, "sequences") else seq
            text_out = self.processor.batch_decode(
                seq[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            gen = (text_out[0] if text_out else "").strip()
        raw = (assistant_prefix + gen) if assistant_prefix else gen
        return {"trajectory": parse_trajectory(raw), "raw": raw,
                "has_foa": has_foa, "n_foa_pad": n_pad}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": "OmniEcho"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/infer":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                parts = req.get("content_parts")
                if not parts:
                    
                    parts = []
                    for p in req.get("images", []):
                        parts.append({"type": "image", "value": p})
                    if req.get("foa"):
                        parts.append({"type": "audio", "value": req["foa"]})
                    if req.get("text"):
                        parts.append({"type": "text", "value": req["text"]})
                out = model.infer_one(
                    parts, max_new_tokens=int(req.get("max_new_tokens", 512)),
                    assistant_prefix=req.get("assistant_prefix") or None)
                self._send(200, out)
            except Exception as e:
                log(f"Inference failed: {e}\n{traceback.format_exc()}")
                self._send(500, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="Inference checkpoint directory containing FOA weights")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8210)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-audio-in-video", action="store_true")
    args = ap.parse_args()

    model = OmniEchoNavModel(
        args.model_path, dtype=args.dtype, attn=args.attn,
        use_audio_in_video=not args.no_audio_in_video)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    log(f"Listening on http://{args.host}:{args.port}  (POST /infer, GET /health)")
    log("READY")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
