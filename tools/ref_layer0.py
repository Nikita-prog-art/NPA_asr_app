#!/usr/bin/env python3
"""Replicate conformer layer 0 in PyTorch straight from the checkpoint and dump
every sub-stage, so that the C++/ggml port can be checked stage by stage.

The input is the golden `subs` tensor ([T, d_model] layout).

Usage: tools/ref_layer0.py IN.npz OUT.npz
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/nikita/projects/NPA_asr_app"
SD = f"{ROOT}/models/hf/gigaam-v3-e2e_rnnt/pytorch_model.bin"


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    d = np.load(src)
    sd = torch.load(SD, map_location="cpu", weights_only=False)

    def g(name: str) -> torch.Tensor:
        return sd["model.encoder.layers.0." + name].float()

    def lin(x, w, b):
        return x @ w.T + b

    x = torch.from_numpy(d["subs"][0]).float()  # [T, D]
    T, D = x.shape
    eps = 1e-5
    # carry every array of the input over, then add the layer-0 sub-stages
    out = {k: d[k] for k in d.files}
    out["l0_in"] = x.numpy()

    # FFN 1
    residual = x
    n = F.layer_norm(x, (D,), g("norm_feed_forward1.weight"), g("norm_feed_forward1.bias"), eps)
    h = lin(F.silu(lin(n, g("feed_forward1.linear1.weight"), g("feed_forward1.linear1.bias"))),
            g("feed_forward1.linear2.weight"), g("feed_forward1.linear2.bias"))
    x = residual + 0.5 * h
    out["l0_ff1"] = x.numpy()

    # self attention: rope applied to the *input* of q/k, NEOX-style pairing
    residual = x
    n = F.layer_norm(x, (D,), g("norm_self_att.weight"), g("norm_self_att.bias"), eps)
    H = 16
    dk = D // H
    nh = n.view(T, H, dk)
    pos = torch.arange(T, dtype=torch.float32)
    inv = 1.0 / (5000.0 ** (torch.arange(0, dk, 2).float() / dk))
    freqs = torch.outer(pos, inv)                     # [T, dk/2]
    emb = torch.cat((freqs, freqs), dim=-1)           # [T, dk]
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    x1, x2 = nh[..., : dk // 2], nh[..., dk // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    nh = nh * cos + rot * sin
    nr = nh.reshape(T, D)
    q = lin(nr, g("self_attn.linear_q.weight"), g("self_attn.linear_q.bias"))
    k = lin(nr, g("self_attn.linear_k.weight"), g("self_attn.linear_k.bias"))
    v = lin(n, g("self_attn.linear_v.weight"), g("self_attn.linear_v.bias"))

    def heads(t):
        return t.view(T, H, dk).permute(1, 0, 2)  # [H, T, dk]

    scores = torch.softmax(heads(q) @ heads(k).transpose(-1, -2) / (dk ** 0.5), dim=-1)
    attn = (scores @ heads(v)).permute(1, 0, 2).reshape(T, D)
    attn = lin(attn, g("self_attn.linear_out.weight"), g("self_attn.linear_out.bias"))
    x = residual + attn
    out["l0_attn"] = x.numpy()

    # convolution module (layer_norm variant)
    residual = x
    n = F.layer_norm(x, (D,), g("norm_conv.weight"), g("norm_conv.bias"), eps)
    c = n.T.unsqueeze(0)  # [1, D, T]
    c = F.conv1d(c, g("conv.pointwise_conv1.weight"), g("conv.pointwise_conv1.bias"))
    c = F.glu(c, dim=1)
    out["l0_glu"] = c.squeeze(0).transpose(0, 1).numpy()
    c = F.conv1d(c, g("conv.depthwise_conv.weight"), g("conv.depthwise_conv.bias"),
                 padding=(5 - 1) // 2, groups=D)
    out["l0_dw"] = c.squeeze(0).transpose(0, 1).numpy()
    c = F.layer_norm(c.transpose(1, 2), (D,), g("conv.batch_norm.weight"), g("conv.batch_norm.bias"), eps)
    out["l0_dwnorm"] = c.squeeze(0).numpy()
    c = F.silu(c).transpose(1, 2)
    c = F.conv1d(c, g("conv.pointwise_conv2.weight"), g("conv.pointwise_conv2.bias"))
    out["l0_pw2"] = c.squeeze(0).transpose(0, 1).numpy()
    x = residual + c.squeeze(0).transpose(0, 1)
    out["l0_conv"] = x.numpy()

    # FFN 2
    residual = x
    n = F.layer_norm(x, (D,), g("norm_feed_forward2.weight"), g("norm_feed_forward2.bias"), eps)
    h = lin(F.silu(lin(n, g("feed_forward2.linear1.weight"), g("feed_forward2.linear1.bias"))),
            g("feed_forward2.linear2.weight"), g("feed_forward2.linear2.bias"))
    x = residual + 0.5 * h
    out["l0_ff2"] = x.numpy()

    x = F.layer_norm(x, (D,), g("norm_out.weight"), g("norm_out.bias"), eps)
    out["layer0"] = x.numpy()
    for k2, v2 in out.items():
        print(f"  {k2}: {np.asarray(v2).shape} min={np.min(v2):.4f} max={np.max(v2):.4f}")
    np.savez_compressed(dst, **out)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
