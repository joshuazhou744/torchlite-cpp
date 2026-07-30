"""
Load a DINOv3 checkpoint .pth file to TLMD
"""

import struct
import torch
import argparse
import os

# write to TLMD
def write_tlmd(path, tensors):
    with open(path, "wb") as f:
        f.write(b"TLMD") # name of the file format (TLMD) in bytes
        f.write(struct.pack("<i", 1)) # < = little endian (LSB), i = signed 4-byte int, indicates version number of TLMD
        f.write(struct.pack("<q", len(tensors))) # little endian signed 8-byte integer (long long), indicates number of tensors in this checkpoint
        for t in tensors: # iterate through all tensors
            t = t.detach().contiguous().float() # detach grad, make contiguous, and cast to float
            f.write(struct.pack("<q", t.ndim)) # dim of the tensor
            for s in t.shape:
                f.write(struct.pack("<q", s)) # size of each dim
            f.write(t.numpy().tobytes()) # raw float32 data of the tensor

# load DINOv3 checkpoint
def load_ckpt(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "cls_token" not in sd: # sanity check if real DINOv3 checkpoint
        raise ValueError("not DINOv3 because no cls_token")
    return sd

# get config of the DINOv3 model (S, B, L)
def infer_config(sd):
    dim = sd["cls_token"].shape[-1]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    n_storage = sd["storage_tokens"].shape[1] if "storage_tokens" in sd else 0

    # patch embed is a conv: [dim, in_channels, k, k]
    pw = sd["patch_embed.proj.weight"]
    patch_size = pw.shape[-1]
    if pw.shape[1] != 3:
        raise ValueError("expected 3 input channels (RGB)")
    if pw.shape[-2] != patch_size:
        raise ValueError("kernel must be square")

    # rope periods hold head_dim // 4
    periods = sd["rope_embed.periods"]
    head_dim = 4 * periods.numel()
    if dim % head_dim != 0:
        raise ValueError("dim must be divisible by rope head_dim")

    num_heads = dim // head_dim

    return {
        "dim": int(dim),
        "depth": int(depth),
        "num_heads": int(num_heads),
        "patch_size": int(patch_size),
        "n_storage": int(n_storage),
    }

# Pytorch stores Linear transposed, we need to transpose it back
def lin(w):
    return w.t().contiguous()

def qkv_bias(sd, prefix):
    return (sd[f"{prefix}.bias"] * sd[f"{prefix}.bias_mask"]).contiguous()

def extract_params(sd, cfg):
    params = []

    params.append(sd["patch_embed.proj.weight"]) # conv [dim, 3, k, k]
    params.append(sd["patch_embed.proj.bias"])
    params.append(sd["cls_token"]) # [1, 1, dim]
    params.append(sd["storage_tokens"]) # [1, n_storage, dim]

    # each DinoBlock: norm1, attn(qkv, proj), ls1, norm2, fc1, fc2, ls2 -> 14 tensors
    for i in range(cfg["depth"]):
        p = f"blocks.{i}"
        params.append(sd[f"{p}.norm1.weight"])
        params.append(sd[f"{p}.norm1.bias"])
        params.append(lin(sd[f"{p}.attn.qkv.weight"]))
        params.append(qkv_bias(sd, f"{p}.attn.qkv"))
        params.append(lin(sd[f"{p}.attn.proj.weight"]))
        params.append(sd[f"{p}.attn.proj.bias"])
        params.append(sd[f"{p}.ls1.gamma"])
        params.append(sd[f"{p}.norm2.weight"])
        params.append(sd[f"{p}.norm2.bias"])
        params.append(lin(sd[f"{p}.mlp.fc1.weight"]))
        params.append(sd[f"{p}.mlp.fc1.bias"])
        params.append(lin(sd[f"{p}.mlp.fc2.weight"]))
        params.append(sd[f"{p}.mlp.fc2.bias"])
        params.append(sd[f"{p}.ls2.gamma"])

    params.append(sd["norm.weight"])
    params.append(sd["norm.bias"])

    expected = 4 + 14 * cfg["depth"] + 2
    assert len(params) == expected, f"{len(params)} tensors, expected {expected}"
    return params

# config file
def write_cfg(path, cfg):
    with open(path, "w") as f:
        for key in ("dim", "depth", "num_heads", "patch_size", "n_storage"):
            f.write(f"{key} {cfg[key]}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="DINOv3 .pth checkpoint")
    ap.add_argument("--out", required=True, help="output file path")
    args = ap.parse_args()

    sd = load_ckpt(args.ckpt)
    cfg = infer_config(sd)
    print("inferred config:", cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    params = extract_params(sd, cfg)
    write_tlmd(f"{args.out}.tl", params)
    nfloats = sum(t.numel() for t in params)
    # 174, 22M params for DINO-S
    # 174, 86M params for DINO-B
    print(f"wrote {len(params)} tensors, {nfloats} params to {args.out}.tl")

    write_cfg(f"{args.out}.cfg", cfg)
    print(f"wrote {args.out}.cfg")

if __name__ == "__main__":
    main()

