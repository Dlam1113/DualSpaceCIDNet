# -*- coding: utf-8 -*-
"""
DualSpaceCIDNet Minimal Inference Demo
Supports evaluation with pretrained weights or random initialized sanity check.
"""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from net.DualSpaceCIDNet import DualSpaceCIDNet


def load_image(image_path):
    """Load and normalize RGB image to float tensor [1, 3, H, W] in range [0, 1]."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def save_image(tensor, save_path):
    """Save normalized float tensor [1, 3, H, W] to image file."""
    tensor = tensor.squeeze(0).clamp(0, 1).detach().cpu()
    arr = (tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    Image.fromarray(arr).save(save_path)


def main():
    parser = argparse.ArgumentParser(description="DualSpaceCIDNet Image Restoration Demo")
    parser.add_argument("--input", type=str, default="demo_images/input.png", help="Path to input degraded image")
    parser.add_argument("--output", type=str, default="demo_images/output.png", help="Path to save restored image")
    parser.add_argument("--weights", type=str, default=None, help="Path to model checkpoint (.pt or .pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[*] Initializing DualSpaceCIDNet on device: {args.device}")
    model = DualSpaceCIDNet().to(args.device)
    model.eval()

    if args.weights and os.path.isfile(args.weights):
        print(f"[*] Loading pretrained weights from: {args.weights}")
        state = torch.load(args.weights, map_location=args.device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    else:
        print("[!] No checkpoint provided or found. Running forward-pass sanity check with initialized weights.")

    if not os.path.exists(args.input):
        print(f"[x] Input image not found: {args.input}")
        return

    print(f"[*] Processing: {args.input}")
    x = load_image(args.input).to(args.device)
    
    # Pad to multiple of 8 for architectural downsampling
    _, _, h, w = x.shape
    pad_h = (-h) % 8
    pad_w = (-w) % 8
    if pad_h > 0 or pad_w > 0:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        x_pad = x

    with torch.no_grad():
        out_pad = model(x_pad)
        out = out_pad[:, :, :h, :w].clamp(0, 1)

    save_image(out, args.output)
    print(f"[OK] Restoration completed! Saved to: {args.output}")


if __name__ == "__main__":
    main()
