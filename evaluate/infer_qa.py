#!/usr/bin/env python
# =============================================================================

# -----------------------------------------------------------------------------


#


#          <foa_bos>(151676) + L*<foa_pad>(151677) + <foa_eos>(151678)


#


#


#


#

#   {index, prediction, reference, media, question, foa, has_foa, n_foa_pad, meta?}

#

# =============================================================================
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from transformers import Qwen3OmniMoeForConditionalGeneration as OmniEchoModel
from transformers import Qwen3OmniMoeProcessor as OmniEchoProcessor
from qwen_omni_utils import process_mm_info

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_ROOT = os.environ.get("OMNIECHO_BENCH_ROOT", os.path.join(REPO_ROOT, "data", "OmniEchoBench"))
DEFAULT_QA_GOLD = os.environ.get("OMNIECHO_QA_GOLD", os.path.join(BENCH_ROOT, "OmniEchoBench-QA", "OmniEchoBench-QA.jsonl"))


_FOA_UTILS_PATH = os.path.join(REPO_ROOT, "omniecho", "foa_features.py")
_spec = importlib.util.spec_from_file_location("foa_encoder_utils", _FOA_UTILS_PATH)
foa_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(foa_utils)
foa_path_to_encoder_features = foa_utils.foa_path_to_encoder_features
foa_num_channels = foa_utils.foa_num_channels


FOA_BOS_ID = int(os.environ.get("OMNIECHO_FOA_BOS_ID", "151676"))
FOA_PAD_ID = int(os.environ.get("OMNIECHO_FOA_PAD_ID", "151677"))
FOA_EOS_ID = int(os.environ.get("OMNIECHO_FOA_EOS_ID", "151678"))
FOA_ENC_DIM = int(os.environ.get("OMNIECHO_FOA_ENC_DIM", "384"))
FOA_PROJ_HIDDEN = int(os.environ.get("OMNIECHO_FOA_PROJ_HIDDEN", "2048"))


_VID_FPS = float(os.environ.get("OMNIECHO_VIDEO_FPS", "2.0"))
_VID_MAX_FRAMES = int(os.environ.get("OMNIECHO_VIDEO_MAX_FRAMES", "16"))
_VID_MAX_PIXELS = int(os.environ.get("OMNIECHO_VIDEO_MAX_PIXELS", str(448 * 448)))


def log(msg: str) -> None:
    print(f"[OmniEcho] {msg}", flush=True)


# =============================================================================

# =============================================================================
def _foa_sinusoidal_pos_emb(length: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class _FOAConvFrontend(nn.Module):
    def __init__(self, in_ch: int = 5, hidden: int = 128, out_ch: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.time_stride = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FOAExternalEncoder(nn.Module):
    def __init__(self, in_ch=5, n_mels=128, conv_hidden=128, conv_out=256, d_model=384,
                 n_layers=5, n_heads=6, ffn=1536, dropout=0.1, max_frames=4096):
        super().__init__()
        self.frontend = _FOAConvFrontend(in_ch, conv_hidden, conv_out)
        freq_out = ((n_mels + 1) // 2 + 1) // 2
        freq_out = (freq_out + 1) // 2
        self.freq_out = freq_out
        self.d_model = d_model
        self.time_stride = self.frontend.time_stride
        self.in_proj = nn.Linear(conv_out * freq_out, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.max_frames = max_frames

    def out_len(self, in_frames: torch.Tensor) -> torch.Tensor:
        n = in_frames
        for _ in range(3):
            n = torch.div(n + 1, 2, rounding_mode="floor")
        return n

    def forward(self, feats: torch.Tensor, frame_lens: torch.Tensor):
        x = self.frontend(feats)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.in_proj(x)
        x = x + _foa_sinusoidal_pos_emb(t, self.d_model, x.device, x.dtype).unsqueeze(0)
        out_lens = self.out_len(frame_lens).clamp(max=t)
        idx = torch.arange(t, device=x.device).unsqueeze(0)
        key_padding_mask = idx >= out_lens.unsqueeze(1)
        key_padding_mask[:, 0] = False
        h = self.transformer(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return h, key_padding_mask


class FOAProjector(nn.Module):
    def __init__(self, enc_dim=384, proj_hidden=2048, hidden=2048):
        super().__init__()
        self.pre_norm = nn.LayerNorm(enc_dim)
        self.fc1 = nn.Linear(enc_dim, proj_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.pre_norm(x))))


def checkpoint_weight_map(ckpt_dir):
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as stream:
            return json.load(stream)["weight_map"]
    shard = "model.safetensors"
    with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as stream:
        return {key: shard for key in stream.keys()}


def load_foa_modules(ckpt_dir, device, dtype, text_hidden):
    """OmniEcho processing component."""
    enc = FOAExternalEncoder(d_model=FOA_ENC_DIM)
    proj = FOAProjector(enc_dim=FOA_ENC_DIM, proj_hidden=FOA_PROJ_HIDDEN, hidden=text_hidden)
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    weight_map = checkpoint_weight_map(ckpt_dir)
    enc_keys = {k: v for k, v in weight_map.items() if k.startswith("thinker.foa_encoder.")}
    proj_keys = {k: v for k, v in weight_map.items() if k.startswith("thinker.foa_projector.")}
    
    def gather(keys, prefix):
        by_shard = {}
        for k, shard in keys.items():
            by_shard.setdefault(shard, []).append(k)
        sd = {}
        for shard, ks in by_shard.items():
            with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
                for k in ks:
                    sd[k[len(prefix):]] = f.get_tensor(k)
        return sd
    enc_sd = gather(enc_keys, "thinker.foa_encoder.")
    proj_sd = gather(proj_keys, "thinker.foa_projector.")
    m1 = enc.load_state_dict(enc_sd, strict=True)
    m2 = proj.load_state_dict(proj_sd, strict=True)
    log(f"foa_encoder load: missing={len(m1.missing_keys)} unexpected={len(m1.unexpected_keys)} "
        f"({len(enc_sd)} tensors)")
    log(f"foa_projector load: missing={len(m2.missing_keys)} unexpected={len(m2.unexpected_keys)} "
        f"({len(proj_sd)} tensors)")
    if m1.missing_keys or m1.unexpected_keys:
        log(f"  enc missing={m1.missing_keys[:4]} unexpected={m1.unexpected_keys[:4]}")
    if m2.missing_keys or m2.unexpected_keys:
        log(f"  proj missing={m2.missing_keys} unexpected={m2.unexpected_keys}")
    enc = enc.to(device=device, dtype=dtype).eval()
    proj = proj.to(device=device, dtype=dtype).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in proj.parameters():
        p.requires_grad_(False)
    return enc, proj


@torch.no_grad()
def compute_foa_embeds(enc, proj, feats_np, L, out_dtype, out_device):
    """OmniEcho processing component."""
    dev = enc.in_proj.weight.device
    feat = torch.from_numpy(feats_np).unsqueeze(0).to(device=dev, dtype=enc.in_proj.weight.dtype)  # (1,5,128,T)
    frame_lens = torch.tensor([feats_np.shape[-1]], dtype=torch.long, device=dev)
    h, kpm = enc(feat, frame_lens)  # (1,T',d), (1,T')
    valid = ~kpm[0]
    v = h[0][valid]
    if v.shape[0] == 0:
        v = h[0][:1]
    v = v.transpose(0, 1).unsqueeze(0)  # (1, d, T'_valid)
    v = F.interpolate(v.float(), size=int(L), mode="linear", align_corners=False)
    v = v.squeeze(0).transpose(0, 1)  # (L, d)
    v = v.to(proj.fc1.weight.dtype)
    emb = proj(v)  # (L, hidden)
    return emb.to(device=out_device, dtype=out_dtype)


# =============================================================================

# =============================================================================
def _clamp_video_params(conversation):
    for turn in conversation:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "video":
                c.setdefault("fps", _VID_FPS)
                c.setdefault("max_frames", _VID_MAX_FRAMES)
                c.setdefault("max_pixels", _VID_MAX_PIXELS)


def _resolve(path, media_root):
    """OmniEcho processing component."""
    if not isinstance(path, str) or not path:
        return path
    if path.startswith(("http://", "https://", "file://", "/")):
        return path
    return os.path.normpath(os.path.join(media_root, path))


REASON_LETTER = (
    "\n\nBefore answering, reason step by step:\n"
    "1) Sound direction: From the 4-channel FOA spatial audio, state the direction the "
    "sound arrives from relative to your forward (left/right, front/back, and a rough "
    "clock direction or angle).\n"
    "2) Your orientation: From your first-person view and the top-down map, determine your "
    "own facing direction and how the top-down map is oriented relative to your forward "
    "(its up direction is not necessarily your forward).\n"
    "3) Reasoning: Map the sound's egocentric direction onto the top-down map and compare "
    "it against the four candidate locations A, B, C and D.\n"
    "Keep the reasoning brief (a few sentences). Then, on the final line, output exactly: "
    "\"Answer: X\" where X is one of A, B, C, D."
)


def split_user_turn(messages, media_root="", reason_instruction=""):
    """OmniEcho processing component."""
    user_turn = None
    reference = None
    for m in messages:
        role = m.get("role")
        if role == "user" and user_turn is None:
            user_turn = m
        elif role == "assistant" and reference is None:
            c = m.get("content")
            reference = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    if user_turn is None:
        raise ValueError("Sample has no user turn")
    content = user_turn.get("content")
    texts, media, foa_src = [], None, None
    if isinstance(content, str):
        texts.append(content)
    else:
        for c in content:
            ctype = c.get("type")
            if ctype == "text":
                texts.append(c.get("text", ""))
            elif ctype in ("image", "video", "audio"):
                path = c.get(ctype) or c.get(f"{ctype}_url")
                path = _resolve(path, media_root)
                
                if c.get(ctype) is not None:
                    c[ctype] = path
                if c.get(f"{ctype}_url") is not None:
                    c[f"{ctype}_url"] = path
                if media is None:
                    media = path
                
                if ctype == "audio":
                    foa_src = path
                elif ctype == "video" and foa_src is None:
                    foa_src = path
    if reason_instruction:
        if isinstance(content, list):
            content.append({"type": "text", "text": reason_instruction})
        else:
            content = str(content) + reason_instruction
        texts.append(reason_instruction)
    conversation = [{"role": "user", "content": content}]
    return conversation, "\n".join(texts).strip(), media, reference, foa_src


def build_inputs_with_foa(processor, conversation, use_audio_in_video, foa_src, vision_end_id):
    """OmniEcho processing component."""
    _clamp_video_params(conversation)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
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
    apad_pos = [i for i, t in enumerate(ids) if t == 151675]  # audio_token_id
    if not apad_pos:
        return inputs, None, None
    L = len(apad_pos)
    aend_idx = apad_pos[-1] + 1  
    insert_at = aend_idx + 1
    if insert_at < len(ids) and ids[insert_at] == vision_end_id:
        insert_at += 1  
    span = [FOA_BOS_ID] + [FOA_PAD_ID] * L + [FOA_EOS_ID]
    ids[insert_at:insert_at] = span
    attn[insert_at:insert_at] = [1] * len(span)
    inputs["input_ids"] = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    inputs["attention_mask"] = torch.tensor(attn, dtype=inputs["attention_mask"].dtype).unsqueeze(0)
    return inputs, L, feats


def _fix_transposed_fused_experts(model, ckpt_dir, dtype):
    """OmniEcho processing component."""
    wm = checkpoint_weight_map(ckpt_dir)
    params = dict(model.named_parameters())
    keys = [k for k in wm if k.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))]
    
    by_file = {}
    for k in keys:
        by_file.setdefault(wm[k], []).append(k)
    n = n_ok = 0
    for fname, ks in by_file.items():
        with safe_open(os.path.join(ckpt_dir, fname), framework="pt", device="cpu") as sf:
            for k in ks:
                if k not in params:
                    log(f"  [warn] Parameter {k} is absent from the model; skipping")
                    continue
                p = params[k]
                ck_shape = tuple(sf.get_slice(k).get_shape())
                if ck_shape == tuple(p.shape):
                    n_ok += 1  
                    continue
                t = sf.get_tensor(k).transpose(1, 2).contiguous()
                if tuple(t.shape) != tuple(p.shape):
                    raise RuntimeError(f"{k}: checkpoint {ck_shape}, transposed {tuple(t.shape)} != model {tuple(p.shape)}")
                with torch.no_grad():
                    p.copy_(t.to(p.device, p.dtype))
                n += 1
    log(f"  Fused expert weights: transposed {n}, already aligned {n_ok}")


def load_done_indices(out_path):
    done = set()
    if not os.path.isfile(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["index"])
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="OmniEcho checkpoint directory containing thinker.foa_* weights")
    ap.add_argument("--test-jsonl", default=DEFAULT_QA_GOLD,
                    help="Benchmark JSONL; defaults to OmniEchoBench-QA/OmniEchoBench-QA.jsonl under OMNIECHO_BENCH_ROOT.")
    ap.add_argument("--media-root", default=None,
                    help="Root for relative media paths (default: directory containing test-jsonl)")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--foa", dest="foa", action="store_true", help="Enable the FOA encoder by inserting and populating an FOA token span")
    ap.add_argument("--no-foa", dest="foa", action="store_false", help="Disable the FOA branch (baseline mode; default)")
    ap.set_defaults(foa=False)
    ap.add_argument("--mute-audio-encoder", dest="mute_audio_encoder", action="store_true",
                    help="Keep the FOA span, but replace the semantic audio tower output at "
                         "audio placeholders (<|AUDIO|> 151675) with the original <|AUDIO|> token "
                         "embedding, retaining only the FOA audio branch when FOA is injected. "
                         "Implicitly enables --foa.")
    ap.set_defaults(mute_audio_encoder=False)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Thinker sampling temperature; >0 enables sampling; 0 leaves checkpoint generation defaults unchanged")
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9,
                    help="Nucleus sampling probability; used only when temperature > 0")
    ap.add_argument("--only-indices", default=None,
                    help="Infer only the original indices listed in a text file; preserve their original index values")
    ap.add_argument("--no-audio-in-video", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--reason", action="store_true",
                    help="Append the cognitive-map reasoning instruction to every selected sample; use only on letter questions")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dtype = getattr(torch, args.dtype)
    use_audio_in_video = not args.no_audio_in_video
    if args.mute_audio_encoder and not args.foa:
        args.foa = True  
    AUDIO_PAD_ID = 151675  

    log(f"Loading model: {args.model_path}  (foa={args.foa}, uaiv={use_audio_in_video})")
    t0 = time.time()
    model = OmniEchoModel.from_pretrained(
        args.model_path, dtype=dtype, device_map="auto", attn_implementation=args.attn,
        ignore_mismatched_sizes=True,
    )


    
    _fix_transposed_fused_experts(model, args.model_path, dtype)
    model.eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    log(f"Model loaded in {time.time() - t0:.1f}s")

    processor = OmniEchoProcessor.from_pretrained(args.model_path)
    vocab = processor.tokenizer.get_vocab()
    vision_end_id = vocab.get("<|vision_end|>")
    text_hidden = model.config.thinker_config.text_config.hidden_size


    stash = {"emb": None, "pos": None, "apos": None}
    enc = proj = None
    if args.foa:
        emb_dev = model.thinker.get_input_embeddings().weight.device
        enc, proj = load_foa_modules(args.model_path, emb_dev, dtype, text_hidden)


        _mute_vec = None
        if args.mute_audio_encoder:
            with torch.no_grad():
                _mute_vec = model.thinker.get_input_embeddings().weight[AUDIO_PAD_ID].detach().clone()

        def _prehook(module, hook_args, hook_kwargs):
            ie = hook_kwargs.get("inputs_embeds", None)
            if ie is None or ie.dim() != 3 or ie.shape[1] <= 1:
                return None  
            emb = stash["emb"]
            pos = stash["pos"]
            apos = stash["apos"]
            has_foa = emb is not None and pos is not None and pos.numel() > 0
            has_mute = _mute_vec is not None and apos is not None and apos.numel() > 0
            if not has_foa and not has_mute:
                return None
            ie = ie.clone()
            
            if has_mute:
                ap = apos.to(ie.device)
                ie[0, ap] = _mute_vec.to(ie.device, ie.dtype)
            
            if has_foa:
                p = pos.to(ie.device)
                ie[0, p] = emb.to(ie.device, ie.dtype)
            hook_kwargs["inputs_embeds"] = ie
            return hook_args, hook_kwargs

        model.thinker.model.register_forward_pre_hook(_prehook, with_kwargs=True)
        log(f"Registered FOA prefill scatter hook (text_hidden={text_hidden}, "
            f"mute_audio_encoder={args.mute_audio_encoder})")

    media_root = args.media_root or os.path.dirname(os.path.abspath(args.test_jsonl))
    log(f"Media root: {media_root}")


    samples = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    if args.num_shards > 1:
        shard = [(i, s) for i, s in enumerate(samples) if i % args.num_shards == args.shard_id]
        log(f"Shard {args.shard_id}/{args.num_shards}: {len(shard)}/{len(samples)}")
    else:
        shard = list(enumerate(samples))
    
    if args.only_indices:
        with open(args.only_indices, encoding="utf-8") as f:
            keep = {int(x) for x in f.read().split() if x.strip()}
        before = len(shard)
        shard = [(i, s) for i, s in shard if i in keep]
        log(f"only-indices filter: {len(shard)}/{before} (requested set: {len(keep)} indices)")

    done = load_done_indices(args.out_jsonl)
    if done:
        log(f"Resuming: skipping {len(done)} recorded indices")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    n_ok = n_err = n_skip = n_foa = 0
    t_start = time.time()
    with open(args.out_jsonl, "a", encoding="utf-8") as fout:
        for idx, sample in shard:
            if idx in done:
                n_skip += 1
                continue
            try:
                conv, question, media, reference, foa_src = split_user_turn(
                    sample["messages"], media_root,
                    reason_instruction=(REASON_LETTER if args.reason else ""))
                inputs, L, feats = (None, None, None)
                if args.foa:
                    inputs, L, feats = build_inputs_with_foa(
                        processor, conv, use_audio_in_video, foa_src, vision_end_id)
                else:
                    inputs, _, _ = build_inputs_with_foa(
                        processor, conv, use_audio_in_video, None, vision_end_id)

                has_foa = 0
                n_pad = 0
                n_mute = 0
                stash["emb"] = stash["pos"] = stash["apos"] = None
                if args.foa and L is not None and feats is not None:
                    ids_t = inputs["input_ids"].squeeze(0)
                    pos = (ids_t == FOA_PAD_ID).nonzero(as_tuple=False).squeeze(-1)
                    n_pad = int(pos.numel())
                    if n_pad > 0:
                        _emb_dev = model.thinker.get_input_embeddings().weight.device
                        emb = compute_foa_embeds(enc, proj, feats, L,
                                                 out_dtype=dtype, out_device=_emb_dev)
                        
                        stash["emb"] = emb
                        stash["pos"] = pos
                        has_foa = 1
                        n_foa += 1
                        
                        if args.mute_audio_encoder:
                            apos = (ids_t == AUDIO_PAD_ID).nonzero(as_tuple=False).squeeze(-1)
                            n_mute = int(apos.numel())
                            stash["apos"] = apos if n_mute > 0 else None

                inputs = inputs.to(model.device).to(model.dtype)
                gen_kwargs = dict(
                    thinker_max_new_tokens=args.max_new_tokens,
                    return_audio=False, use_audio_in_video=use_audio_in_video,
                )
                if args.temperature and args.temperature > 0:
                    
                    gen_kwargs.update(
                        thinker_do_sample=True,
                        thinker_temperature=args.temperature,
                        thinker_top_p=args.top_p,
                    )
                with torch.no_grad():
                    out_ids = model.generate(**inputs, **gen_kwargs)
                seq = out_ids[0] if isinstance(out_ids, (tuple, list)) else out_ids
                seq = seq.sequences if hasattr(seq, "sequences") else seq
                text_out = processor.batch_decode(
                    seq[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                pred = (text_out[0] if text_out else "").strip()
                stash["emb"] = stash["pos"] = stash["apos"] = None  

                rec = {
                    "index": idx, "prediction": pred, "reference": reference,
                    "media": media, "question": question,
                    "foa": bool(args.foa), "has_foa": has_foa, "n_foa_pad": n_pad,
                    "mute_audio_encoder": bool(args.mute_audio_encoder), "n_audio_muted": n_mute,
                }
                if "meta" in sample:
                    rec["meta"] = sample["meta"]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
                if n_ok <= 5 or n_ok % 25 == 0:
                    rate = n_ok / max(time.time() - t_start, 1e-6)
                    log(f"[{idx+1}] ok (has_foa={has_foa} n_pad={n_pad} n_mute={n_mute} {rate:.2f}it/s) pred={pred[:70]!r}")
            except Exception as e:
                n_err += 1
                stash["emb"] = stash["pos"] = stash["apos"] = None
                log(f"[{idx+1}] Failed: {e}")
                traceback.print_exc()
                fout.write(json.dumps({"index": idx, "error": str(e)}, ensure_ascii=False) + "\n")
                fout.flush()

    log(f"Done: ok={n_ok} err={n_err} skip={n_skip} foa_injected={n_foa}  elapsed={time.time()-t_start:.1f}s -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
