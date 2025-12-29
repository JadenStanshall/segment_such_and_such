import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import SegmentationDataset
from model import UNet

def train(
    data_root,
    results_dir,
    epochs,
    batch_size,
    lr
):
    os.makedirs(results_dir, exist_ok=True)
    log_dir = os.path.join(results_dir, "runs")
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = SegmentationDataset(
        images_dir=os.path.join(data_root, "train/images"),
        masks_dir =os.path.join(data_root, "train/masks"),
        augment=True
    )
    val_ds = SegmentationDataset(
        images_dir=os.path.join(data_root, "val/images"),
        masks_dir =os.path.join(data_root, "val/masks"),
        augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # model, loss, optimizer
    model = UNet().to("cpu")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Train E{epoch}"):
            imgs, masks = imgs.to("cpu"), masks.to("cpu")
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to("cpu"), masks.to("cpu")
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_loader)

        # logs
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # checkpoints
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"unet_epoch{epoch}.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "unet_best.pth"))

    writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a UNet segmentation model")
    p.add_argument("--data-root",
                   default="/home/jaden/projects/sew/segment/pytorch/data",
                   help="root directory of train/val folders")
    p.add_argument("--results-dir",
                   default="results",
                   help="where to write checkpoints and logs")
    p.add_argument("--epochs", type=int, default=25,
                   help="number of training epochs")
    p.add_argument("--batch-size", type=int, default=4,
                   help="batch size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="learning rate")
    args = p.parse_args()

    train(
        data_root   = args.data_root,
        results_dir = args.results_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr
    )
