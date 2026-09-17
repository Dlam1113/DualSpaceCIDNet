# DualSpaceCIDNet

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper:  
**"Cascaded Dual-Space Color-Illumination Decoupling Network with Neural Curve Adjustment for All-Weather Image Restoration"**

---

## 📌 Method Overview

![Overall Architecture](figures/overall_architecture.png)

DualSpaceCIDNet proposes a cascaded color-illumination decoupling framework tailored for challenging adverse-weather and composite degradation conditions (including low light, dense haze, heavy rain, snow, and mixed nighttime corruptions). 

### Key Highlights:
1. **Continuous Polarized HVI Decoupling**: Mathematically maps coupled non-linear sRGB into continuous Cartesian chromaticity $(H, V)$ and illumination intensity $(I)$, eliminating color-space singularities and dark-plane noise inherent in cylindrical color representations.
2. **Adaptive Neural Curve Adjustment**: Introduces an 11-control-point piecewise-linear curve module on the intensity branch to perform dynamic illumination mapping.
3. **Output-Domain Residual Refiner**: Implements an end-to-end zero-initialized residual refiner operating directly in the output RGB domain to learn high-fidelity spatial details.

---

## 🛠️ Environment Setup

Clone this repository and set up the Python environment:

```bash
git clone https://github.com/Dlam1113/DualSpaceCIDNet.git
cd DualSpaceCIDNet
pip install -r requirements.txt
```

---

## 🚀 Quick Start (Inference Demo)

We provide a lightweight standalone inference script `demo.py` to evaluate degraded images:

```bash
# Run sanity-check inference on sample image
python demo.py --input demo_images/input.png --output demo_images/output.png
```

To evaluate with a trained checkpoint:
```bash
python demo.py --input demo_images/input.png --output demo_images/output.png --weights weights/best_model.pt
```

---

## 📊 Benchmark Results

Evaluated on the official **CDD-11 Adverse-Weather Benchmark** (2,200 testing image pairs across 11 single and composite degradation conditions):

| Model | All-11 Average (PSNR / SSIM / LPIPS) | Triple-Composite (PSNR / SSIM / LPIPS) | Parameters | FLOPs (256x256) |
| :--- | :---: | :---: | :---: | :---: |
| PromptIR | 25.24 / 0.8198 / 0.1620 | 22.01 / 0.7180 / 0.2310 | 35.59 M | 347.47 G |
| NAFNet | 24.15 / 0.7944 / 0.2202 | 21.48 / 0.6774 / 0.3433 | 17.11 M | 32.44 G |
| Baseline CIDNet | 24.78 / 0.8320 / 0.1450 | 22.26 / 0.7226 / 0.2286 | 1.84 M | 19.82 G |
| **DualSpaceCIDNet (Ours)** | **24.53 / 0.8344 / 0.1442** | **22.40 / 0.7287 / 0.2261** | **2.02 M** | **21.01 G** |

*DualSpaceCIDNet achieves state-of-the-art restoration on extreme composite weather conditions while requiring only 2.02M parameters (5.7% of PromptIR).*

---

## 📦 Training Code and Checkpoint Release Notice

- **Pretrained Weights**: The full trained checkpoints for all benchmarks will be released on cloud storage immediately upon formal acceptance of the manuscript.
- **Training Pipeline & Datasets**: The complete multi-weather dataset pipelines, manifest generators, and multi-GPU distributed training scripts are currently being packaged and will be fully open-sourced upon paper publication.

---

## 📑 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{jin2026dualspacecidnet,
  title   = {Cascaded Dual-Space Color-Illumination Decoupling Network with Neural Curve Adjustment for All-Weather Image Restoration},
  author  = {Jin, Si-Nian and Bao, Junjun and Ju, Moran},
  journal = {Neurocomputing},
  year    = {2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
