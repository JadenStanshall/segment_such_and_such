import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        # encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = n_channels
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = f

        # bottleneck
        self.bottleneck = DoubleConv(in_ch, in_ch * 2)
        in_ch = in_ch * 2

        # decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2))
            # after up-sample, channels double because of concat
            self.up_convs.append(DoubleConv(f * 2, f))
            in_ch = f

        # outpu layer
        self.final_conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # down path
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # bottleneck
        x = self.bottleneck(x)

        # up path
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skip_connections)):
            x = up(x)
            # fix potential size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final_conv(x)
