# DRF-Net: Dual-Stream Residual Fusion Network for Face Morphing Attack Detection

Official implementation of the paper:

**A Novel Dual-stream Residual Fusion Network (DRF-Net) for Accurate Detection of Face Morphing Attack in Biometric Systems**

---

## Overview

DRF-Net is a dual-stream deep learning framework for detecting face morphing attacks in biometric systems. The model combines handcrafted texture descriptors with two encoder-decoder networks (Xception and ResNet50) to learn both semantic facial features and reconstruction residuals.

The framework is designed to identify subtle morphing artefacts that are difficult to detect using conventional deep learning approaches.

---

## Key Features

- Dual-stream architecture
- Xception U-Net branch
- ResNet50 U-Net branch
- Residual map generation
- Feature-level fusion
- MLP classifier
- Face morphing attack detection

---

## Method Pipeline

```
Input Image
      │
Preprocessing
      │
LBP + HOG + BSIF
      │
──────────────────────────
│                        │
▼                        ▼
UNet-Xception      UNet-ResNet50
│                        │
Residual Maps      Residual Maps
│                        │
──────── Feature Fusion ────────
              │
      MLP Classifier
              │
     Morph / Bona fide
```

---

## Repository Structure

```
.
├── DRF-NET.py
├── Face_Morphing.pdf
├── images/
├── models/
└── README.md
```

---

## Performance

| Metric | Value |
|---------|-------|
| Accuracy | **98.69%** |
| Precision | **98.69%** |
| Recall | **98.69%** |
| F1-score | **98.69%** |
| BPCER | **0.0156** |
| EER | **0.0119** |

---

## Paper

The complete research paper is available in this repository:

**Face_Morphing.pdf**

---

## Future Work

- Cross-dataset evaluation
- Real-time morph detection
- Video-based morph detection
- Lightweight deployment

---

## Author

Your Name

Department / University
