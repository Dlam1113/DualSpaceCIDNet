# -*- coding: utf-8 -*-
"""
Transformer and Convolution Utilities.
Provides channel-wise LayerNorm, downsampling, and upsampling modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""
    Layer Normalization supporting channels_first and channels_last data formats.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        """
        Forward pass of LayerNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


class NormDownsample(nn.Module):
    """
    Normalized downsampling block using convolution followed by bilinear interpolation.
    """
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )

    def forward(self, x):
        """
        Downsamples spatial dimensions while transforming feature channels.

        Args:
            x (torch.Tensor): Input feature map [B, in_ch, H, W].

        Returns:
            torch.Tensor: Downsampled feature map [B, out_ch, H/2, W/2].
        """
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
        return x


class NormUpsample(nn.Module):
    """
    Normalized upsampling block with skip-connection concatenation and channel reduction.
    """
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        """
        Upsamples decoder feature map and fuses it with encoder skip-connection feature.

        Args:
            x (torch.Tensor): Low-resolution decoder feature [B, in_ch, H/2, W/2].
            y (torch.Tensor): High-resolution skip-connection feature [B, out_ch, H, W].

        Returns:
            torch.Tensor: Fused feature map [B, out_ch, H, W].
        """
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        return x
