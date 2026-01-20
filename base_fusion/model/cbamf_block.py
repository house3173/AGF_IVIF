import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max_ = F.adaptive_max_pool2d(x, 1)
        return torch.sigmoid(self.mlp(avg) + self.mlp(max_))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_], dim=1)
        return torch.sigmoid(self.conv(x))

class CBAMFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels * 2)
        self.sa = SpatialAttention()

        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, f_ir, f_vi):
        # Concatenate
        f = torch.cat([f_ir, f_vi], dim=1)

        # CBAM
        f = f * self.ca(f)
        f = f * self.sa(f)

        # Mix
        f = self.conv(f)

        # Split weights
        w = self.softmax(f)
        w_ir, w_vi = torch.chunk(w, 2, dim=1)

        # Weighted fusion
        f_fused = w_ir * f_ir + w_vi * f_vi
        return f_fused
