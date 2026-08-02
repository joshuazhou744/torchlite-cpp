import os
import sys
import torch
import argparse

from loader import load_ckpt, infer_config, write_tlmd


def build_reference(cfg, ckpt_path, dino_root):
    sys.path.insert(0, os.path.expanduser(dino_root)) # make sure dinov3 is cloned somewhere on this machine
    from dinov3.hub import backbones

    variants = {
        (384, 12): backbones.dinov3_vits16,
        (768, 12): backbones.dinov3_vitb16,
        (1024, 24): backbones.dinov3_vitl16,
    }

    key = (cfg["dim"], cfg["depth"])
    if key not in variants:
        raise ValueError(f"unknown variant for dim={key[0]} depth={key[1]}")

    model = variants[key](weights=ckpt_path)
    model.eval()
    return model

def run_inference(model, batch, height, width, layers, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, 3, height, width)
    with torch.no_grad():
        feats = model.get_intermediate_layers(x, n=layers, reshape=True, norm=True)
    return x, list(feats)

def write_golden_cfg(path, batch, height, width, layers):
    with open(path, "w") as f:
        f.write(f"batch {batch}\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("layers " + " ".join(str(l) for l in layers) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--height", type=int, default=112)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--dino-root", default="~/repos/dinov3")
    args = ap.parse_args()

    sd = load_ckpt(args.ckpt)
    cfg = infer_config(sd)

    layers = args.layers if args.layers else [0, cfg["depth"] // 2, cfg["depth"] - 1]
    layers = sorted(set(layers))
    for l in layers:
        if not 0 <= l < cfg["depth"]:
            raise ValueError(f"layer {l} out of range for depth {cfg['depth']}")
    for name, v in (("height", args.height), ("width", args.width)):
        if v % cfg["patch_size"] != 0:
            raise ValueError(f"{name} {v} not divisible by patch_size {cfg['patch_size']}")

    model = build_reference(cfg, args.ckpt, args.dino_root)
    x, feats = run_inference(model, args.batch, args.height, args.width, layers)

    write_tlmd(f"{args.out}.golden", [x] + feats)
    write_golden_cfg(f"{args.out}.golden.cfg", args.batch, args.height, args.width, layers)
    print(f"wrote {args.out}.golden")

if __name__ == "__main__":
    main()
