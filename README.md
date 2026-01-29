# BrandGuard — Counterfeit Logo Detection (GenAI Synthetic Data + ResNet18)

BrandGuard is an end-to-end pipeline for **detecting counterfeit logos** using:
1) **Synthetic counterfeit generation** (SDXL Img2Img + post-defects)  
2) **Binary classification** (ResNet18: Real vs Fake)

The focus is intentionally **logo-level** (not “the whole shoe”), so the model learns *logo semantics* (shape/typography/placement cues) instead of overfitting to backgrounds.

---

## Project Motivation
Real-world counterfeit datasets are limited, noisy, and hard to label.  
So instead of hunting for “perfect fake data”, we generate **controlled, realistic counterfeits** and train a classifier that can generalize across **different backgrounds, materials, and image quality**.

---

## Problem statement
Given an input image that contains a brand logo (preferably a crop/close-up), classify it as:
- **Real**: authentic logo
- **Fake**: counterfeit-like distortions (typos, warped geometry, missing strokes, printing artifacts, etc.)

---

## Visual abstract
![Workflow Example](results/workflow_example.png)

---

## Repository structure
```text
BrandGuard/
├── data_generation/
│   ├── generate_fakes.py          # SDXL Img2Img + counterfeit defects
│
├── model_training/
│   ├── train_classifier.py        # ResNet18 training + evaluation
│
├── results/                       # Images and Graphs
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── dataset_hierarchy.png
│   ├── workflow_example.png
│   └── examples_strip.png
│
└── README.md                      # Project documentation
```
## Datasets used or collected

### Real (Authentic)
Source: `original_logo/<BRAND>/...`  
Includes:
- original logo images (logo-only / close-up)
- **real augmentations**:
  - small rotations
  - brightness changes
  - **low-quality simulation** (blur + noise + down/upscale artifacts)

> Why include low-quality *Real* images?  
> We found the model tends to learn a shortcut: **“low quality = fake”**.  
> So we inject low-quality Real samples to force the model to learn *logo features*, not image quality.

### Fake (Synthetic Counterfeit)
Source: generated into `counterfeit/<BRAND>/...`  
Generated with:
- **SDXL Img2Img** (`stabilityai/stable-diffusion-xl-base-1.0`)
- brand-specific photorealistic prompts
- two regimes:
  - **LOW strength** (subtle counterfeits)
  - **HIGH strength + typo forcing** (aggressive counterfeits)

Dataset composition example:  
![Dataset Hierarchy](results/dataset_hierarchy.png)

---

## Data augmentation and generation methods

### 1) Real augmentations (non-fake)
Implemented in `generate_authentic_variations()`:
- `rotate(-5..5)`
- `brightness(0.95..1.05)`
- `make_image_low_quality()` (blur + noise + resize artifacts)

Output goes to:
- `real_augmented/<BRAND>/...` (recommended separation from originals)

### 2) Fake generation (GenAI)
Implemented in `generate_ai_fakes()`:
- SDXL Img2Img runs on a resized 768×768 version of the input logo
- Uses brand prompt templates (Converse/Nike/Adidas/...)
- Chooses randomly between:
  - subtle mode: `strength ~ 0.45–0.55`
  - aggressive mode: `strength ~ 0.70–0.88` + typo forcing with probability `TYPO_PROB`

### 3) Post-defects (classic image ops)
After SDXL generates a fake, we apply one random **post defect** to simulate printing/manufacturing errors:
- thin / thicken strokes
- missing parts
- double print
- color shift
- blur edges
- **perspective warp**
- outline ring
- **edge jitter**
- missing-letter style removal

This is what gives you both:
- “looks almost real but off”
- and “clearly wrong counterfeit”

Example strip:  
![Examples Strip](results/examples_strip.png)

---

## Workflow visualization
1. Put authentic logo images into: `original_logo/<BRAND>/`
2. Run `generate_fakes.py`
   - generates **Real augmentations** into `real_augmented/<BRAND>/`
   - generates **Fake counterfeits** into `counterfeit/<BRAND>/`
3. Run `train_classifier.py`
   - builds a train folder: `/content/training_data/{Real,Fake}`
   - trains **ResNet18** on Real vs Fake
   - saves model: `models/logo_resnet18.pth`
   - outputs plots + confusion matrix

---

## Training process and parameters
Training code is in `model_training/train_classifier.py`.

### Data preparation
- Copies everything from:
  - `original_logo/**` → `training_data/Real`
  - `counterfeit/**`   → `training_data/Fake`

### Transforms
Train:
- Resize 256 → CenterCrop 224
- RandomHorizontalFlip
- RandomRotation(10)
- ColorJitter(brightness=0.3, contrast=0.3)
- RandomGrayscale(p=0.3)
- Normalize (ImageNet mean/std)

Val:
- Resize 256 → CenterCrop 224
- Normalize

### Hyperparameters
- Split: 80/20 (stratified)
- Batch size: 16
- Epochs: 15
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropy + class weights (handles imbalance)

---

## Results
Accuracy / Loss curves:
![Accuracy](results/accuracy_plot.png)
![Loss](results/loss_plot.png)

Confusion matrix:
![Confusion Matrix](results/confusion_matrix.png)

---

## How to run

### 1) Generate synthetic dataset
```bash
python data_generation/generate_fakes.py
