import torch
import torch.nn as nn
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

class MyUNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MyUNet, self).__init__()
        self.ReLU = torch.nn.ReLU(inplace=True)
        self.Sigmoid = torch.nn.Sigmoid()
        
        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.btn = DoubleConv(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)  
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc1_max = nn.MaxPool2d(kernel_size=2, stride=2)(enc1)

        enc2 = self.encoder2(enc1_max)
        enc2_max = nn.MaxPool2d(kernel_size=2, stride=2)(enc2)
        
        enc3 = self.encoder3(enc2_max)
        enc3_max = nn.MaxPool2d(kernel_size=2, stride=2)(enc3)
        
        enc4 = self.encoder4(enc3_max)
        enc4_max = nn.MaxPool2d(kernel_size=2, stride=2)(enc4)

        # Bottleneck
        bottleneck = self.btn(enc4_max)
        dec1 = self.up1(bottleneck)

        # Decoder
        
        # layer 4
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.decoder4(dec1)
        # layer 3
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.decoder3(dec2)
        # layer 2
        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder2(dec3)
        # layer 1
        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.decoder1(dec4)
        # Final Convolution
        return self.final_conv(dec4)
