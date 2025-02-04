import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution: (Conv -> ReLU -> Conv -> ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Down(nn.Module):
    """
    Down-sampling with maxpool then double conv.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """
    Up-sampling (either by transposed conv or bilinear upsample) then double conv.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: feature map from previous up/initial
        x2: skip connection from encoder
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    """
    Basic UNet architecture:
    - 5 levels (4 downs, center, 4 ups)
    - Each level uses a DoubleConv
    - Skip connections from down to up
    """

    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # For feature map extraction, we can store them in a dictionary
        self.features = {}

    def forward(self, x):
        self.features = {}  # reset/clear at each forward

        x1 = self.inc(x)
        self.features['enc1'] = x1

        x2 = self.down1(x1)
        self.features['enc2'] = x2

        x3 = self.down2(x2)
        self.features['enc3'] = x3

        x4 = self.down3(x3)
        self.features['enc4'] = x4

        x5 = self.down4(x4)
        self.features['enc5'] = x5

        x = self.up1(x5, x4)
        self.features['dec1'] = x

        x = self.up2(x, x3)
        self.features['dec2'] = x

        x = self.up3(x, x2)
        self.features['dec3'] = x

        x = self.up4(x, x1)
        self.features['dec4'] = x

        logits = self.outc(x)
        self.features['out'] = logits

        return logits


if __name__ == "__main__":
    # Simple test
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
    print("Feature maps:")
    for k, v in model.features.items():
        print(k, v.shape)
