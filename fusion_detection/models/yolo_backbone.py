# models/yolo_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels//2, 1),
            ConvBlock(channels//2, channels, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        
        # Downsample 1
        self.conv2 = ConvBlock(32, 64, 3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(64)
        
        # Downsample 2
        self.conv3 = ConvBlock(64, 128, 3, stride=2, padding=1)
        self.res_blocks2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])
        
        # Downsample 3
        self.conv4 = ConvBlock(128, 256, 3, stride=2, padding=1)
        self.res_blocks3 = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])
        
        # Downsample 4
        self.conv5 = ConvBlock(256, 512, 3, stride=2, padding=1)
        self.res_blocks4 = nn.Sequential(*[ResidualBlock(512) for _ in range(8)])
        
        # Downsample 5
        self.conv6 = ConvBlock(512, 1024, 3, stride=2, padding=1)
        self.res_blocks5 = nn.Sequential(*[ResidualBlock(1024) for _ in range(4)])
        
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # First block and downsample
        x = self.conv2(x)
        x = self.res_block1(x)
        
        # Second block and downsample
        x = self.conv3(x)
        x = self.res_blocks2(x)
        
        # Third block and downsample
        x = self.conv4(x)
        x = self.res_blocks3(x)
        route_1 = x  # First route connection
        
        # Fourth block and downsample
        x = self.conv5(x)
        x = self.res_blocks4(x)
        route_2 = x  # Second route connection
        
        # Fifth block and downsample
        x = self.conv6(x)
        x = self.res_blocks5(x)
        route_3 = x  # Third route connection
        
        return route_1, route_2, route_3

class YOLOBackbone(nn.Module):
    def __init__(self, num_classes=8):
        super(YOLOBackbone, self).__init__()
        self.num_classes = num_classes
        
        # Darknet53 backbone
        self.darknet = Darknet53()
        
        # Output convolutions for detection at different scales
        # Large scale (detect largest objects)
        self.conv_large = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1)
        )
        
        # Medium scale (detect medium objects)
        self.conv_medium = nn.Sequential(
            ConvBlock(768, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1)
        )
        
        # Small scale (detect smallest objects)
        self.conv_small = nn.Sequential(
            ConvBlock(384, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 128, 1)
        )
        
        # Upsampling
        self.upsample1 = nn.Sequential(
            ConvBlock(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        self.upsample2 = nn.Sequential(
            ConvBlock(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # Prediction layers
        self.pred_large = nn.Conv2d(512, 3 * (5 + num_classes), 1)
        self.pred_medium = nn.Conv2d(256, 3 * (5 + num_classes), 1)
        self.pred_small = nn.Conv2d(128, 3 * (5 + num_classes), 1)
        
    def forward(self, x):
        # Features from Darknet backbone
        route_1, route_2, route_3 = self.darknet(x)
        
        # Large scale detection
        large_feat = self.conv_large(route_3)
        large_output = self.pred_large(large_feat)
        
        # Upsample and concatenate for medium scale
        upsampled_large = self.upsample1(large_feat)
        medium_input = torch.cat([upsampled_large, route_2], dim=1)
        medium_feat = self.conv_medium(medium_input)
        medium_output = self.pred_medium(medium_feat)
        
        # Upsample and concatenate for small scale
        upsampled_medium = self.upsample2(medium_feat)
        small_input = torch.cat([upsampled_medium, route_1], dim=1)
        small_feat = self.conv_small(small_input)
        small_output = self.pred_small(small_feat)
        
        # Return features and predictions
        return {
            'features': {
                'large': large_feat,
                'medium': medium_feat,
                'small': small_feat
            },
            'predictions': [large_output, medium_output, small_output]
        }
