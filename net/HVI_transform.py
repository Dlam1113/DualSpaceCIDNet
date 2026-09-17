# -*- coding: utf-8 -*-
"""
Continuous Polarized HVI Color Space Transformation Module.
Provides differentiable forward (RGB -> HVI) and inverse (HVI -> RGB) color space transformations.
"""
import torch
import torch.nn as nn

PI = 3.141592653589793


class RGB_HVI(nn.Module):
    """
    Differentiable RGB to Polarized HVI (Hue-Value-Intensity) transformation module.
    Decouples coupled RGB space into continuous chromaticity (H, V) and illumination intensity (I).
    """
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = nn.Parameter(torch.full([1], 0.2))  # Learnable color sensitivity density exponent
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0.2

    def HVIT(self, img):
        """
        Forward transformation from sRGB to HVI space.

        Args:
            img (torch.Tensor): Input sRGB image tensor of shape [B, 3, H, W] in range [0, 1].

        Returns:
            torch.Tensor: Transformed HVI representation of shape [B, 3, H, W].
        """
        eps = 1e-8
        device = img.device
        dtypes = img.dtype

        hue = torch.zeros(img.shape[0], img.shape[2], img.shape[3], device=device, dtype=dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)

        # Standard HSV hue angle formulation
        mask_b = (img[:, 2] == value)
        mask_g = (img[:, 1] == value)
        mask_r = (img[:, 0] == value)

        hue[mask_b] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[mask_b]
        hue[mask_g] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[mask_g]
        hue[mask_r] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[mask_r]) % 6.0
        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        # Saturation component
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0.0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        # Polar projection to continuous Cartesian chromaticity coordinates (H, V)
        color_sensitive = ((value * 0.5 * PI).sin() + eps).pow(k)
        ch = (2.0 * PI * hue).cos()
        cv = (2.0 * PI * hue).sin()

        h_channel = color_sensitive * saturation * ch
        v_channel = color_sensitive * saturation * cv
        i_channel = value

        return torch.cat([h_channel, v_channel, i_channel], dim=1)

    def PHVIT(self, img):
        """
        Inverse transformation from HVI back to sRGB space.

        Args:
            img (torch.Tensor): HVI representation tensor of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Reconstructed sRGB image tensor of shape [B, 3, H, W].
        """
        eps = 1e-8
        h_channel, v_channel, i_channel = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # Clamp components within theoretical bounds
        h_channel = torch.clamp(h_channel, -1.0, 1.0)
        v_channel = torch.clamp(v_channel, -1.0, 1.0)
        i_channel = torch.clamp(i_channel, 0.0, 1.0)

        v = i_channel
        k = self.this_k
        color_sensitive = ((v * 0.5 * PI).sin() + eps).pow(k)

        # Recover normalized trigonometric coordinates
        h_norm = h_channel / (color_sensitive + eps)
        v_norm = v_channel / (color_sensitive + eps)
        h_norm = torch.clamp(h_norm, -1.0, 1.0)
        v_norm = torch.clamp(v_norm, -1.0, 1.0)

        h = torch.atan2(v_norm + eps, h_norm + eps) / (2.0 * PI)
        h = h % 1.0
        s = torch.sqrt(h_norm ** 2 + v_norm ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))

        hi0 = (hi == 0)
        hi1 = (hi == 1)
        hi2 = (hi == 2)
        hi3 = (hi == 3)
        hi4 = (hi == 4)
        hi5 = (hi == 5)

        r[hi0], g[hi0], b[hi0] = v[hi0], t[hi0], p[hi0]
        r[hi1], g[hi1], b[hi1] = q[hi1], v[hi1], p[hi1]
        r[hi2], g[hi2], b[hi2] = p[hi2], v[hi2], t[hi2]
        r[hi3], g[hi3], b[hi3] = p[hi3], q[hi3], v[hi3]
        r[hi4], g[hi4], b[hi4] = t[hi4], p[hi4], v[hi4]
        r[hi5], g[hi5], b[hi5] = v[hi5], p[hi5], q[hi5]

        rgb = torch.cat([r.unsqueeze(1), g.unsqueeze(1), b.unsqueeze(1)], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
