import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)
    
# Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             ConvBlock(channels, channels, kernel_size=3, padding=1),
#             ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
#         )
    
#     def forward(self, x):
#         return x + self.block(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + checkpoint(self.block, x)

# Generator
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9, text_feat_dim=512):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),

            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        
        self.text_proj_small = nn.Linear(512, 256 * 4 * 4)
        
        
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4 + 256, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        
    def forward(self, x, text_feat):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x= self.residual_blocks(x)
        
                # text_feat shape: (B, 512)
        B = x.size(0)

        # Project and reshape text features to match up_block feature map shapes
        
        text_feat = text_feat.float()
        
        text_embed = self.text_proj_small(text_feat).view(B, 256, 4, 4)
        text_embed = F.interpolate(text_embed, size=(64, 64), mode='bilinear')
        x = torch.cat([x, text_embed], dim=1)
        
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def test():
    img_channels =3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)
    
if __name__ == '__main__':
    test()