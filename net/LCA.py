# -*- coding: utf-8 -*-
"""
Lightweight Cross Attention (LCA) and Intensity Enhancement Modules.
Enables bidirectional cross-attention feature interaction between chromaticity (HV) and intensity (I) streams.
"""
import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import LayerNorm


class CAB(nn.Module):
    """
    Cross Attention Block (CAB).
    Allows a query feature map (Q) from one stream to attend to key-value representations (K, V) from the other.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query projection and depthwise spatial mixing
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Key-Value projection and depthwise spatial mixing
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        Computes cross-attention between query feature x and context feature y.

        Args:
            x (torch.Tensor): Query feature tensor of shape [B, C, H, W].
            y (torch.Tensor): Key/Value feature tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Attended feature tensor of shape [B, C, H, W].
        """
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        # Reshape for multi-head attention over spatial locations
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize features onto unit hypersphere for stable direction-based attention
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class IEL(nn.Module):
    """
    Intensity Enhancement Layer (IEL).
    Applies gated non-linear feedforward transformations to modulate feature dynamics.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass with dual-branch gated non-linear modulation.

        Args:
            x (torch.Tensor): Input feature tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Modulated feature tensor of shape [B, C, H, W].
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.tanh(self.dwconv1(x1)) + x1
        x2 = self.tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class HV_LCA(nn.Module):
    """
    Lightweight Cross-Attention Block for the Chromaticity (HV) Stream.
    Allows HV features to query guidance information from the Intensity (I) stream.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.gdfn = IEL(dim)

    def forward(self, x, y):
        """
        Cross-attention update for HV stream.

        Args:
            x (torch.Tensor): HV feature tensor [B, C, H, W].
            y (torch.Tensor): I feature tensor [B, C, H, W].

        Returns:
            torch.Tensor: Enhanced HV feature tensor [B, C, H, W].
        """
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    """
    Lightweight Cross-Attention Block for the Intensity (I) Stream.
    Allows I features to query context information from the Chromaticity (HV) stream.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)

    def forward(self, x, y):
        """
        Cross-attention update for I stream with dual residual pathways.

        Args:
            x (torch.Tensor): I feature tensor [B, C, H, W].
            y (torch.Tensor): HV feature tensor [B, C, H, W].

        Returns:
            torch.Tensor: Enhanced I feature tensor [B, C, H, W].
        """
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x
