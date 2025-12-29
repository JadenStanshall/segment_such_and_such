import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False, img_size=(368,224)):
        exts = ('.jpg', '.jpeg', '.png')
        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(exts)])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

        base_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(), # [0,255]→[0,1]
        ]
        aug_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
        self.img_transform = transforms.Compose(
            (aug_transforms + base_transforms) if augment else base_transforms
        )
        self.mask_transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fn = self.images[idx]
        img = Image.open(os.path.join(self.images_dir, fn)).convert("RGB")
        base, _ = os.path.splitext(fn)
        mask_fn = base + '.png'
        m = Image.open(os.path.join(self.masks_dir, mask_fn)).convert("L")

        # sync RNG
        seed = np.random.randint(2**32)
        random.seed(seed); img = self.img_transform(img)
        random.seed(seed); m   = self.mask_transform(m)

        # free space=1, background/obstacles=0
        m = (m > 0.5).float()

        return img, m
