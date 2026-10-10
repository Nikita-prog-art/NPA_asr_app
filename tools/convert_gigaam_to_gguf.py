#!/usr/bin/env python3
"""Convert the GigaAM-v3 `e2e_rnnt` checkpoint (HF ai-sage/GigaAM-v3, revision e2e_rnnt)
into a GGUF file that the ggml/llama.cpp ecosystem can read.

Design notes
------------
* Tensors are stored with the *same numpy shapes as PyTorch*: gguf-py writes the
  dimensions reversed, so a numpy array of shape (out, in) becomes a ggml tensor
  with ne = (in, out) - exactly what `ggml_mul_mat(w, x)` wants.  Conv1d weights
  (out_ch, in_ch, k) become ggml (k, in_ch, out_ch), i.e. `ggml_conv_1d` layout.
  No transposes are needed anywhere except for the mel filterbank.
* The mel filterbank is stored as (n_mels, n_fft/2+1) so that ggml ne is
  (n_fft/2+1, n_mels) and `ggml_mul_mat(fb, power_spectrogram)` yields mel bands.
  The 320-point Hann window is stored verbatim.
* The SentencePiece vocabulary is embedded as an array of strings, so the runtime
  needs no `tokenizer.model` file.

Usage:
  convert_gigaam_to_gguf.py --model-dir DIR --out FILE [--quant f16|q8_0|q5_0|q4_0]
  convert_gigaam_to_gguf.py --list-tensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType, LlamaFileType
from gguf.quants import quantize

ARCH = "gigaam"
QUANTS = {
    "f32": GGMLQuantizationType.F32,   # reference build, for numerical checks
    "f16": GGMLQuantizationType.F16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q4_0": GGMLQuantizationType.Q4_0,
}
FILE_TYPES = {
    "f32": LlamaFileType.ALL_F32,
    "f16": LlamaFileType.MOSTLY_F16,
    "q8_0": LlamaFileType.MOSTLY_Q8_0,
    "q5_0": LlamaFileType.MOSTLY_Q5_0,
    "q4_0": LlamaFileType.MOSTLY_Q4_0,
}

# ---------------------------------------------------------------- name mapping

LAYER_RENAMES = {
    "feed_forward1.linear1": "ff1.linear1",
    "feed_forward1.linear2": "ff1.linear2",
    "feed_forward2.linear1": "ff2.linear1",
    "feed_forward2.linear2": "ff2.linear2",
    "self_attn.linear_q": "attn.q",
    "self_attn.linear_k": "attn.k",
    "self_attn.linear_v": "attn.v",
    "self_attn.linear_out": "attn.out",
    "conv.pointwise_conv1": "conv.pw1",
    "conv.depthwise_conv": "conv.dw",
    "conv.pointwise_conv2": "conv.pw2",
    "conv.batch_norm": "conv.norm",
    "norm_feed_forward1": "norm_ff1",
    "norm_feed_forward2": "norm_ff2",
    "norm_self_att": "norm_att",
    "norm_conv": "norm_conv",
    "norm_out": "norm_out",
}

TOP_RENAMES = {
    "preprocessor.featurizer.0.spectrogram.window": "mel.window",
    "head.decoder.embed.weight": "dec.embed.weight",
    "head.decoder.lstm.weight_ih_l0": "dec.lstm.w_ih",
    "head.decoder.lstm.weight_hh_l0": "dec.lstm.w_hh",
    "head.decoder.lstm.bias_ih_l0": "dec.lstm.b_ih",
    "head.decoder.lstm.bias_hh_l0": "dec.lstm.b_hh",
}

RE_LAYER = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")
RE_SUBS = re.compile(r"^encoder\.pre_encode\.conv\.(\d+)\.(.+)$")


def map_name(key: str) -> str:
    if key.startswith("model."):
        key = key[len("model.") :]
    if key == "preprocessor.featurizer.0.mel_scale.fb":
        return "mel.fb"
    m = RE_SUBS.match(key)
    if m:
        return f"subs.conv{m.group(1)}.{m.group(2)}"
    m = RE_LAYER.match(key)
    if m:
        blk, rest = m.group(1), m.group(2)
        for src, dst in LAYER_RENAMES.items():
            if rest.startswith(src + "."):
                return f"blk.{int(blk)}.{dst}.{rest[len(src) + 1:]}"
        raise KeyError(f"unmapped encoder layer tensor: {key}")
    if key in TOP_RENAMES:
        return TOP_RENAMES[key]
    for src, dst in (
        ("head.joint.enc.", "joint.enc."),
        ("head.joint.pred.", "joint.pred."),
        ("head.joint.joint_net.1.", "joint.out."),
    ):
        if key.startswith(src):
            return dst + key[len(src) :]
    raise KeyError(f"unmapped tensor: {key}")


def is_quantizable(name: str, arr: np.ndarray) -> bool:
    """Only 2D matmul weights whose fastest dim is a multiple of 32."""
    if arr.ndim != 2:
        return False
    if name.startswith("mel."):
        return False
    return arr.shape[1] % 32 == 0


def keep_f32(name: str) -> bool:
    """The decoder/joint and the mel front-end are fp32 in the original
    checkpoint, so keep them exact.  The encoder is fp16 natively, hence f16
    storage is lossless for it."""
    return name.startswith(("dec.", "joint.", "mel."))


# ---------------------------------------------------------------- vocab


def read_vocab(model_dir: Path) -> list[str]:
    from sentencepiece import SentencePieceProcessor

    sp = SentencePieceProcessor()
    if not sp.load(str(model_dir / "tokenizer.model")):
        raise RuntimeError("cannot load tokenizer.model")
    return [sp.id_to_piece(i) for i in range(sp.get_piece_size())]


# ---------------------------------------------------------------- conversion


def build_writer(model_dir: Path, cfg: dict, quant: str, out: Path) -> tuple[GGUFWriter, dict]:
    inner = cfg["cfg"]["model"]["cfg"]
    enc = inner["encoder"]
    pre = inner["preprocessor"]
    dec = inner["head"]["decoder"]
    joint = inner["head"]["joint"]
    vocab = read_vocab(model_dir)

    w = GGUFWriter(str(out), arch=ARCH)
    w.add_name(f"GigaAM-v3 {inner['model_name']}")
    w.add_string(f"{ARCH}.model_name", inner["model_name"])
    # front-end
    w.add_uint32(f"{ARCH}.sample_rate", pre["sample_rate"])
    w.add_uint32(f"{ARCH}.n_mels", pre["features"])
    w.add_uint32(f"{ARCH}.n_fft", pre["n_fft"])
    w.add_uint32(f"{ARCH}.win_length", pre["win_length"])
    w.add_uint32(f"{ARCH}.hop_length", pre["hop_length"])
    w.add_bool(f"{ARCH}.center", bool(pre.get("center", False)))
    # encoder
    w.add_uint32(f"{ARCH}.encoder.n_layers", enc["n_layers"])
    w.add_uint32(f"{ARCH}.encoder.d_model", enc["d_model"])
    w.add_uint32(f"{ARCH}.encoder.n_heads", enc["n_heads"])
    w.add_uint32(f"{ARCH}.encoder.d_ff", enc["d_model"] * enc["ff_expansion_factor"])
    w.add_uint32(f"{ARCH}.encoder.conv_kernel", enc["conv_kernel_size"])
    w.add_uint32(f"{ARCH}.encoder.subs_kernel", enc["subs_kernel_size"])
    w.add_uint32(f"{ARCH}.encoder.subsampling_factor", enc["subsampling_factor"])
    w.add_uint32(f"{ARCH}.encoder.pos_emb_max_len", enc["pos_emb_max_len"])
    w.add_uint32(f"{ARCH}.encoder.rope_base", enc["pos_emb_max_len"])
    w.add_float32(f"{ARCH}.encoder.norm_eps", 1e-5)
    w.add_string(f"{ARCH}.encoder.attn", enc["self_attention_model"])
    w.add_string(f"{ARCH}.encoder.conv_norm", enc["conv_norm_type"])
    # decoder / joint
    w.add_uint32(f"{ARCH}.decoder.pred_hidden", dec["pred_hidden"])
    w.add_uint32(f"{ARCH}.decoder.pred_layers", dec["pred_rnn_layers"])
    w.add_uint32(f"{ARCH}.decoder.num_classes", dec["num_classes"])
    w.add_uint32(f"{ARCH}.decoder.blank_id", dec["num_classes"] - 1)
    w.add_uint32(f"{ARCH}.decoder.joint_hidden", joint["joint_hidden"])
    w.add_uint32(f"{ARCH}.decoder.max_symbols_per_step", 10)
    w.add_uint32(f"{ARCH}.vocab_size", len(vocab))
    # tokenizer
    w.add_string("tokenizer.ggml.model", "gigaam_spm")
    w.add_array("tokenizer.ggml.tokens", vocab)
    w.add_file_type(int(FILE_TYPES[quant].value))
    w.add_quantization_version(2)
    return w, {"vocab": vocab, "quant": quant}


def convert(model_dir: Path, out: Path, quant: str) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    w, info = build_writer(model_dir, cfg, quant, out)
    qtype = QUANTS[quant]
    n_quant = 0
    total = 0
    for key in sorted(sd.keys()):
        name = map_name(key)
        t = sd[key]
        if not hasattr(t, "numpy"):
            raise TypeError(f"{key}: not a tensor ({type(t)})")
        arr = t.detach().to(torch.float32).numpy()
        if name == "mel.fb":
            arr = np.ascontiguousarray(arr.T)  # (n_mels, n_freq) -> ggml ne=(n_freq, n_mels)
        total += arr.size
        if quant == "f32" or keep_f32(name):
            w.add_tensor(name, arr)
        elif quant != "f16" and is_quantizable(name, arr):
            qarr = quantize(arr, qtype)
            w.add_tensor(name, qarr.view(np.uint8), raw_shape=arr.shape, raw_dtype=qtype)
            n_quant += 1
        else:
            w.add_tensor(name, arr.astype(np.float16))
    print(f"[convert] tensors: {len(sd)}  params: {total/1e6:.1f}M  quantized: {n_quant}  quant={quant}")

    out.parent.mkdir(parents=True, exist_ok=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    data = out.read_bytes()
    info.update(
        {
            "file": out.name,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "params": total,
        }
    )
    print(f"[convert] wrote {out} ({len(data)/1e6:.1f} MB) sha256={info['sha256'][:16]}...")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/hf/gigaam-v3-e2e_rnnt"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quant", default="f16", choices=sorted(QUANTS))
    ap.add_argument("--list-tensors", action="store_true")
    ap.add_argument("--manifest", type=Path, help="write/update a JSON manifest entry")
    ap.add_argument("--release-base", default="", help="URL prefix for the download link")
    args = ap.parse_args()

    if args.list_tensors:
        sd = torch.load(args.model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        for key in sorted(sd.keys()):
            print(f"{map_name(key):45s} <- {key}")
        return 0

    out = args.out or (args.model_dir / f"gigaam-v3-e2e_rnnt-{args.quant}.gguf")
    info = convert(args.model_dir, out, args.quant)

    if args.manifest:
        manifest = {"schema": 1, "models": []}
        if args.manifest.exists():
            manifest = json.loads(args.manifest.read_text())
        entry = next(
            (m for m in manifest["models"] if m["id"] == "gigaam-v3-e2e_rnnt"),
            None,
        )
        if entry is None:
            entry = {
                "id": "gigaam-v3-e2e_rnnt",
                "name": "GigaAM-v3 e2e_rnnt",
                "repo": "https://huggingface.co/ai-sage/GigaAM-v3",
                "revision": "e2e_rnnt",
                "license": "MIT",
                "files": [],
            }
            manifest["models"].append(entry)
        entry["files"] = [f for f in entry["files"] if f["quant"] != args.quant]
        entry["files"].append(
            {
                "quant": args.quant,
                "url": f"{args.release_base.rstrip('/')}/{out.name}" if args.release_base else out.name,
                "size": info["size"],
                "sha256": info["sha256"],
                "params": info["params"],
            }
        )
        entry["files"].sort(key=lambda f: -f["size"])
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"[convert] manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
