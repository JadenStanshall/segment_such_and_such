import os
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from model import UNet


def infer(
    image_path: str,
    checkpoint: str,
    device: str,
    threshold: float = 0.5,
    resize_hw: tuple[int, int] = (368, 224),
):
    # load model
    model = UNet().to(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}. "
            "Provide --checkpoint or copy your trained unet_best.pth into pytorch/src/checkpoints/."
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # load image
    img = Image.open(image_path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize(resize_hw),
        transforms.ToTensor(),
    ])
    resized_img = tf(img)
    inp = resized_img.unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        pred = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
    mask = (pred > threshold).astype(np.uint8)

    # visualization array
    rgb = resized_img.permute(1, 2, 0).cpu().numpy()
    overlay = rgb.copy()
    overlay[mask == 1] = np.array([0, 1, 0])  # green

    # plots
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (Resized)")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def _default_checkpoint() -> str:
    # Default: pytorch/src/checkpoints/unet_best.pth (relative to this file)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "checkpoints", "unet_best.pth")


def main() -> int:
    p = argparse.ArgumentParser(description="Run UNet segmentation inference on a single image")
    p.add_argument("image", help="Path to an input image (jpg/png)")
    p.add_argument(
        "--checkpoint",
        default=_default_checkpoint(),
        help="Path to a .pth checkpoint (defaults to pytorch/src/checkpoints/unet_best.pth)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Mask threshold (after sigmoid)")
    p.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(368, 224),
        metavar=("H", "W"),
        help="Resize input to H W before inference",
    )
    args = p.parse_args()

    try:
        infer(
            image_path=args.image,
            checkpoint=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
            resize_hw=(args.resize[0], args.resize[1]),
        )
    except Exception as e:
        print(f"Inference failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
