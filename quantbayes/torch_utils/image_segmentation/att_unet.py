"""
attention_unet.py

An updated Attention U-Net in PyTorch that avoids shape mismatches
by properly upsampling the gating signal before combining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class AttentionGate(nn.Module):
    """
    A simple attention gate that uses a gating signal (often from the decoder)
    to focus on relevant features in the skip connection (from the encoder).
    We'll upsample 'g' to match 'x' spatial size, then do 1x1 convs so shapes match.
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(AttentionGate, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        # 1x1 conv to reduce skip feature dimensions
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        # 1x1 conv to reduce gating feature dimensions
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)

        # Combine and map to 1 channel for sigmoid
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """
        x: skip connection feature (B, in_channels, H, W)
        g: gating feature (B, gating_channels, H', W')
           Typically H' < H, W' < W, but we will upsample g.
        """
        # 1) Down-channel skip
        theta_x = self.theta(x)  # shape: (B, inter_channels, H, W)

        # 2) Upsample gating to match skip's spatial size
        g_upsampled = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        phi_g = self.phi(g_upsampled)  # shape: (B, inter_channels, H, W)

        # 3) Add and activate
        f = self.relu(theta_x + phi_g)

        # 4) Map to 1 channel and take sigmoid
        psi_f = self.sigmoid(self.psi(f))

        # 5) Apply attention: multiply original skip by attention mask
        return x * psi_f


class Down(nn.Module):
    """
    Downsampling by maxpool + double conv.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class UpAtt(nn.Module):
    """
    Up-sampling, then apply an attention gate to the skip connection,
    then concatenate, then double conv.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpAtt, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # If using transposed conv, in_channels//2 => the intended # of filters
            self.up = nn.ConvTranspose2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=2,
                stride=2
            )

        # The attention gate:
        #  - skip has in_channels//2
        #  - gating has in_channels//2
        # So we pass in_channels//2 for both
        self.att_gate = AttentionGate(in_channels // 2, in_channels // 2)

        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: feature from the previous decoder layer (gating)
            shape: (B, in_channels//2, H', W')
        x2: skip connection from the encoder
            shape: (B, in_channels//2, H, W)
        """
        # 1) Upsample the gating signal
        x1 = self.up(x1)  # shape => (B, in_channels//2, H, W) if perfect alignment

        # 2) Attention gate: refine skip with gating
        x2_att = self.att_gate(x2, x1)

        # 3) If there's any misalignment, pad x1
        diffY = x2_att.size()[2] - x1.size()[2]
        diffX = x2_att.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 4) Concatenate
        x = torch.cat([x2_att, x1], dim=1)

        # 5) Double conv
        x = self.double_conv(x)
        return x


class AttentionUNet(nn.Module):
    """
    An Attention U-Net that uses attention gates on the skip connections.
    """

    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # bottom layer

        # Decoder with Attention
        self.up1 = UpAtt(1024, 512 // factor, bilinear)
        self.up2 = UpAtt(512, 256 // factor, bilinear)
        self.up3 = UpAtt(256, 128 // factor, bilinear)
        self.up4 = UpAtt(128, 64, bilinear)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # For debugging or feature extraction
        self.features = {}

    def forward(self, x):
        # Reset features each forward
        self.features = {}

        # Encoder
        x1 = self.inc(x)        # (B,64,H,W)
        x2 = self.down1(x1)     # (B,128,H/2,W/2)
        x3 = self.down2(x2)     # (B,256,H/4,W/4)
        x4 = self.down3(x3)     # (B,512,H/8,W/8)
        x5 = self.down4(x4)     # (B,1024//factor,H/16,W/16) if bilinear= True => 512 channels

        self.features['enc1'] = x1
        self.features['enc2'] = x2
        self.features['enc3'] = x3
        self.features['enc4'] = x4
        self.features['enc5'] = x5

        # Decoder (with attention gates)
        # up1 expects in_channels=1024 => gating is x5, skip is x4
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
    model = AttentionUNet(n_channels=3, n_classes=2, bilinear=True)
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)
    print("Output shape:", out.shape)
    for name, tensor in model.features.items():
        print(name, tensor.shape)
