# utils/generate_masks.py
#!/usr/bin/env python3
import os, argparse, cv2, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

def build_mask_generator(model_type: str, ckpt: str, device: str) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
    return SamAutomaticMaskGenerator(sam)

def gen_masks(samgen, img_path, out_base):
    img = cv2.imread(img_path)
    masks = samgen.generate(img)
    name = os.path.splitext(os.path.basename(img_path))[0]
    dst = os.path.join(out_base, name)
    os.makedirs(dst, exist_ok=True)
    for i, m in enumerate(masks):
        mask = (m["segmentation"] * 255).astype("uint8")
        cv2.imwrite(f"{dst}/{name}_mask{i:03d}.png", mask)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", default="sam_masks")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Set to 'cpu' to avoid CUDA OOM on small GPUs (default: auto-detect).")
    p.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                   help="SAM model variant. Ensure checkpoint matches the type.")
    args = p.parse_args()

    device = resolve_device(args.device)
    samgen = build_mask_generator(args.model_type, args.ckpt, device)

    imgs = sorted(f for f in os.listdir(args.in_dir)
                  if f.lower().endswith((".jpg","png","jpeg")))
    for fn in tqdm(imgs, desc="Don't worry, I'm working. Generating SAM masks"):
        try:
            gen_masks(samgen,
                      os.path.join(args.in_dir, fn),
                      args.out_dir)
        except RuntimeError as e:
            # Gracefully fall back to CPU if we run out of GPU memory mid-run.
            if "out of memory" in str(e).lower() and device == "cuda":
                print("CUDA OOM encountered, retrying on CPU...")
                torch.cuda.empty_cache()
                device = "cpu"
                samgen = build_mask_generator(args.model_type, args.ckpt, device)
                gen_masks(samgen,
                          os.path.join(args.in_dir, fn),
                          args.out_dir)
            else:
                raise

if __name__ == "__main__":
    main()
