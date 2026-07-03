import struct
import torch

def write_tl(path, tensors):
    with open(path, 'wb') as f:
        f.write(b'TLMD') # magic
        f.write(struct.pack('<i', 1)) # version int32
        f.write(struct.pack('<q', len(tensors))) # num_tensors int64
        for t in tensors:
            t = t.contiguous().float()
            f.write(struct.pack('<q', t.ndim))
            for s in t.shape:
                f.write(struct.pack('<q', s))
            f.write(t.numpy().tobytes()) # raw float32 data

def load_pt(path):
    return torch.load(path, map_location="cpu")

def inspect_pt(path):
    sd = load_pt(path)
    for k, v in sd.items():
        print(k, v.shape)

def t(sd, key):
    # non-linear tensor: no transpose needed (conv, embedding, buffer, groupnorm)
    return sd[key]

def lin(sd, key):
    # linear weight: pytorch stores [out, in], torchlite stores [in, out]
    return sd[key].t().contiguous()

def resblock_params(sd, prefix, should_proj, has_attn):
    params = []
    if should_proj:
        params.append(t(sd, f"{prefix}.proj.weight"))
        params.append(t(sd, f"{prefix}.proj.bias"))
    params.append(lin(sd, f"{prefix}.norm1.linear.weight"))
    params.append(t(sd, f"{prefix}.norm1.linear.bias"))
    params.append(t(sd, f"{prefix}.conv1.weight"))
    params.append(t(sd, f"{prefix}.conv1.bias"))
    params.append(lin(sd, f"{prefix}.norm2.linear.weight"))
    params.append(t(sd, f"{prefix}.norm2.linear.bias"))
    params.append(t(sd, f"{prefix}.conv2.weight"))
    params.append(t(sd, f"{prefix}.conv2.bias"))
    if has_attn:
        params.append(t(sd, f"{prefix}.attn.norm.norm.weight"))
        params.append(t(sd, f"{prefix}.attn.norm.norm.bias"))
        params.append(t(sd, f"{prefix}.attn.qkv_proj.weight"))
        params.append(t(sd, f"{prefix}.attn.qkv_proj.bias"))
        params.append(t(sd, f"{prefix}.attn.out_proj.weight"))
        params.append(t(sd, f"{prefix}.attn.out_proj.bias"))
    return params


def extract_inner_model_params(sd, cfg):
    prefix = "denoiser.inner_model"
    params = []

    # noise_emb, act_emb, cond_proj, conv_in
    params.append(t(sd, f"{prefix}.noise_emb.weight"))
    params.append(t(sd, f"{prefix}.act_emb.0.weight"))
    params.append(lin(sd, f"{prefix}.cond_proj.0.weight"))
    params.append(t(sd, f"{prefix}.cond_proj.0.bias"))
    params.append(lin(sd, f"{prefix}.cond_proj.2.weight"))
    params.append(t(sd, f"{prefix}.cond_proj.2.bias"))
    params.append(t(sd, f"{prefix}.conv_in.weight"))
    params.append(t(sd, f"{prefix}.conv_in.bias"))

    depths = cfg["depths"]
    channels = cfg["channels"]
    attn_depths = cfg["attn_depths"]
    num_levels = len(depths)

    # encoder: d_blocks
    for i in range(num_levels):
        n = depths[i]
        c1 = channels[max(0, i - 1)]
        c2 = channels[i]
        attn = bool(attn_depths[i])
        for j in range(n):
            should_proj = (j == 0) and (c1 != c2)
            params.extend(resblock_params(sd, f"{prefix}.unet.d_blocks.{i}.resblocks.{j}", should_proj, attn))

    # decoder: u_blocks, stored deepest first (reversed) in both pytorch and our c++ build order
    for i in range(num_levels):
        orig_level = num_levels - 1 - i  # reversed index maps back to original level
        n = depths[orig_level]
        attn = bool(attn_depths[orig_level])
        # decoder resblocks always have proj (channel counts never match after cat)
        for j in range(n + 1):
            params.extend(resblock_params(sd, f"{prefix}.unet.u_blocks.{i}.resblocks.{j}", True, attn))

    # mid_blocks: 2 resblocks, in=out=channels[-1], attn=True, no proj
    for j in range(2):
        params.extend(resblock_params(sd, f"{prefix}.unet.mid_blocks.resblocks.{j}", False, True))

    # downsamples: indices 1..num_down (index 0 is identity, skipped)
    num_down = num_levels - 1
    for i in range(1, num_down + 1):
        params.append(t(sd, f"{prefix}.unet.downsamples.{i}.conv.weight"))
        params.append(t(sd, f"{prefix}.unet.downsamples.{i}.conv.bias"))

    # upsamples: indices 1..num_down (index 0 is identity, skipped)
    for i in range(1, num_down + 1):
        params.append(t(sd, f"{prefix}.unet.upsamples.{i}.conv.weight"))
        params.append(t(sd, f"{prefix}.unet.upsamples.{i}.conv.bias"))

    # norm_out, conv_out
    params.append(t(sd, f"{prefix}.norm_out.norm.weight"))
    params.append(t(sd, f"{prefix}.norm_out.norm.bias"))
    params.append(t(sd, f"{prefix}.conv_out.weight"))
    params.append(t(sd, f"{prefix}.conv_out.bias"))

    return params


if __name__ == "__main__":
    path = "models/Breakout.pt"
    cfg = {
        "depths": [2, 2, 2, 2],
        "channels": [64, 64, 64, 64],
        "attn_depths": [0, 0, 0, 0],
    }

    sd = load_pt(path)
    params = extract_inner_model_params(sd, cfg)
    write_tl("models/Breakout.tl", params)
    print(f"wrote {len(params)} tensors to models/Breakout.tl")
