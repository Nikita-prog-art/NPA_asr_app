#!/usr/bin/env python3
"""PyTorch reference pipeline for GigaAM-v3 e2e_rnnt.

Builds the *official* model code (from .ref/modeling_gigaam.py, downloaded from
the HF repo) on top of the local state dict, runs it on a wav file and dumps
intermediate tensors as goldens for the C++/ggml implementation.

Usage:
  tools/ref_gigaam.py <wav> [--dump OUT.npz] [--dump-layers] [--text-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / ".ref"
sys.path.insert(0, str(REF))

import omegaconf  # noqa: E402

import modeling_gigaam as mg  # noqa: E402


def load_model(model_dir: Path, dtype=torch.float32):
    cfg_json = json.loads((model_dir / "config.json").read_text())
    cfg = omegaconf.OmegaConf.create(cfg_json["cfg"]["model"]["cfg"])
    cfg.decoding.model_path = str(model_dir / "tokenizer.model")
    model = mg.GigaAMASR(cfg)
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ref] missing={missing[:5]} unexpected={unexpected[:5]}", file=sys.stderr)
    model = model.eval().to(dtype)
    return model, cfg


def greedy_tokens(model, encoded, seqlen, max_symbols=10):
    """Same algorithm as modeling_gigaam.RNNTGreedyDecoding._greedy_decode."""
    head = model.head
    blank = model.decoding.blank_id
    x = encoded.transpose(1, 2)[0].unsqueeze(1)  # [D, 1, T]
    hyp = []
    dec_state = None
    last_label = None
    for t in range(seqlen):
        f = x[t, :, :].unsqueeze(1)
        not_blank = True
        new_symbols = 0
        while not_blank and new_symbols < max_symbols:
            g, hidden = head.decoder.predict(last_label, dec_state)
            k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
            if k == blank:
                not_blank = False
            else:
                hyp.append(int(k))
                dec_state = hidden
                last_label = torch.tensor([[hyp[-1]]])
                new_symbols += 1
    return hyp


def run(model, wav_path: Path, dump_layers: bool = False):
    out = {}
    fe = model.preprocessor.featurizer[0]
    out["mel_fb"] = fe.mel_scale.fb.detach().cpu().numpy()          # [1+n_fft/2, n_mels]
    out["mel_window"] = fe.spectrogram.window.detach().cpu().numpy()  # [win_length]
    wav = mg.load_audio(str(wav_path)).unsqueeze(0)  # [1, N]
    out["wav"] = wav.numpy()
    with torch.inference_mode():
        feats, feat_len = model.preprocessor(wav, torch.full([1], wav.shape[-1]))
        out["mel"] = feats.numpy()
        out["mel_len"] = np.asarray([int(feat_len)])
        enc = None
        if dump_layers:
            # manual encoder pass to capture per-layer activations
            x = feats.transpose(1, 2)
            x, length = model.encoder.pre_encode(x, feat_len)
            out["subs"] = x.numpy()
            if not hasattr(model.encoder.pos_enc, "pe"):
                model.encoder.pos_enc.extend_pe(model.encoder.pos_emb_max_len, x.device)
            max_len = x.size(1)
            x, pos_emb = model.encoder.pos_enc(x)
            for li, layer in enumerate(model.encoder.layers):
                x = layer(x=x, pos_emb=pos_emb, att_mask=None, pad_mask=None)
                out[f"layer{li}"] = x.numpy()
            enc = x.transpose(1, 2)
            enc_len = length
        else:
            enc, enc_len = model.encoder(feats, feat_len)
        out["enc"] = enc.numpy()
        out["enc_len"] = np.asarray([int(enc_len)])
        # token ids of the greedy decode (mirrors RNNTGreedyDecoding but returns ids)
        # joint logits of the very first greedy step (debug golden)
        with torch.inference_mode():
            f0 = enc.transpose(1, 2)[0, 0, :].reshape(1, 1, -1)
            g0, _ = model.head.decoder.predict(None, None)
            # raw (pre log_softmax) joint output plus the predictor state
            j = model.head.joint
            z = j.joint_net[0](j.enc(f0) + j.pred(g0))
            out["logits0_raw"] = j.joint_net[1](z).reshape(-1).numpy()
            e0 = j.enc(f0).reshape(-1)
            p0 = j.pred(g0).reshape(-1)
            out["encproj0"] = e0.numpy()
            out["predproj0"] = p0.numpy()
            out["jointin0"] = torch.relu(e0 + p0).numpy()
            out["g0"] = g0.reshape(-1).numpy()
            out["logits0"] = j.joint(f0, g0).reshape(-1).numpy()
        toks = greedy_tokens(model, enc, int(enc_len))
        out["tokens"] = np.asarray(toks, dtype=np.int64)
        text = model.decoding.tokenizer.decode(toks)
    return out, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model-dir", default=str(ROOT / "models/hf/gigaam-v3-e2e_rnnt"))
    ap.add_argument("--dump")
    ap.add_argument("--dump-layers", action="store_true")
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()

    model, cfg = load_model(Path(args.model_dir))
    out, text = run(model, Path(args.wav), dump_layers=args.dump_layers)
    print(f"[ref] mel {out['mel'].shape} enc {out['enc'].shape} enc_len {out['enc_len']}")
    print(f"[ref] text: {text}")
    if args.dump and not args.text_only:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump, **out)
        print(f"[ref] wrote {args.dump}")


if __name__ == "__main__":
    main()
