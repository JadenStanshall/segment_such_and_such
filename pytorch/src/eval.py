import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from model import UNet

def iou_score(preds, masks, thresh):
    preds = (preds > thresh).float()
    inter = (preds * masks).sum((1,2))
    union = ((preds + masks) > 0).float().sum((1,2))
    return (inter / union).mean().item()

def evaluate(
    data_root: str,
    checkpoint: str,
    batch_size: int,
    thresh: float,
    device: str
):
    # prepare dataset
    test_ds = SegmentationDataset(
        images_dir=os.path.join(data_root, "test/images"),
        masks_dir =os.path.join(data_root, "test/masks"),
        augment=False
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # run eval
    ious = []
    with torch.no_grad():
        for img, mask in tqdm(loader, desc="Eval"):
            img, mask = img.to(device), mask.to(device)
            pred = torch.sigmoid(model(img)).squeeze(1)  # [B,H,W]
            ious.append(iou_score(pred, mask.squeeze(1), thresh))
    mean_iou = np.mean(ious)
    print(f"Mean IoU on test set: {mean_iou:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate UNet on a test set")
    p.add_argument("--data-root",
                   default="data",
                   help="root directory of test/images & test/masks")
    p.add_argument("--checkpoint",
                   default="checkpoints/unet_best.pth",
                   help="path to saved .pth checkpoint")
    p.add_argument("--batch-size", type=int, default=1,
                   help="batch size for evaluation")
    p.add_argument("--thresh", type=float, default=0.5,
                   help="sigmoid threshold for IoU")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="device to run on (cuda or cpu)")
    args = p.parse_args()

    evaluate(
        data_root  = args.data_root,
        checkpoint = args.checkpoint,
        batch_size = args.batch_size,
        thresh     = args.thresh,
        device     = args.device
    )
