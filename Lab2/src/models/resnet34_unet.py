import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        return self.relu(x)

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super().__init__()
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MyResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MyResNet34UNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # ResNet34 layers: [3, 4, 6, 3]
        self.layer1 = ResNetLayer(64, 64, 3, stride=1)
        self.layer2 = ResNetLayer(64, 128, 4, stride=2)
        self.layer3 = ResNetLayer(128, 256, 6, stride=2)
        self.layer4 = ResNetLayer(256, 512, 3, stride=2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.dec1 = DoubleConv(512, 256)
        self.dec2 = DoubleConv(256, 128)
        self.dec3 = DoubleConv(128, 64)
        self.dec4 = DoubleConv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1p = self.pool1(x1)
        x2 = self.layer1(x1p)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # Decoder with skip connections
        d1 = self.up1(x5)
        if d1.shape[2:] != x4.shape[2:]:
            d1 = F.interpolate(d1, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        if d2.shape[2:] != x3.shape[2:]:
            d2 = F.interpolate(d2, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        if d4.shape[2:] != x1.shape[2:]:
            d4 = F.interpolate(d4, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.dec4(d4)

        out = self.final(d4)
        return out
        