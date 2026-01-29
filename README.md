# Counterfeit Logo Detection (Synthetic Data + CNN Classifier)

## Project Motivation
Counterfeit products are everywhere, but *reliable labeled counterfeit image datasets* are rare.  
So we built our own pipeline: generate realistic counterfeit logo variants with GenAI, then train a classifier to detect **Real vs Fake** logos robustly across different backgrounds.

---

## Problem statement
Given an image containing a **brand logo** (cropped area from a product photo), classify it as:
- **Real** (authentic logo appearance)
- **Fake** (counterfeit-like distortions: wrong typography, warped shape, missing parts, bad printing, etc.)

Goal: learn the *logo semantics* (shape/typography) and not “cheat” by relying on artifacts like image quality or background.

---

## Visual abstract
![Visual Abstract](Screenshot%202026-01-29%20165318.png)

---

## Datasets used or collected
We created a **custom dataset** because “real counterfeit” data is limited/unreliable.

### Real (Authentic)
- Manually collected **logo-only images / logo crops** (multiple brands).
- Added realistic variations to prevent overfitting (angle/light/quality).

### Fake (Synthetic Counterfeit)
- Generated using **Stable Diffusion XL (Img2Img)** + post-processing defects.
- Includes both **subtle** and **aggressive** counterfeits (typos + distortions).

### Dataset composition (example)
![Dataset Composition](Screenshot%202026-01-29%20165346.png)

---

## Data augmentation and generation methods

### A) Real augmentations (class: Real)
We expanded each authentic logo into many “realistic captures”:
- slight rotation, brightness changes
- **low quality simulation** (blur + resize artifacts + noise)  
This specifically solved a failure mode where the model learned: **“low quality = fake”**.

### B) Fake generation (class: Fake)
**GenAI stage (SDXL Img2Img):**
- Generates photorealistic “counterfeit-like” variants
- Two modes:
  - **LOW strength**: subtle counterfeits
  - **HIGH strength + typo variants**: aggressive counterfeits (e.g., `NYKE`, `ABIBAS`, etc.)

**Post-defects stage (PIL/Numpy):**
- perspective/warp
- outline ring
- edge jitter
- thinning / thickening logo strokes
- missing region / missing “letter-like” parts
- double-print effect
- color shifts, blurry edges

---

## Workflow visualization
1. **Collect** a small set of authentic logo images (per brand)
2. **Generate Real augmentations** (lighting/angle/quality)
3. **Generate Fake images** (SDXL Img2Img + random defects)
4. **Build dataset folders**: `Real/` and `Fake/`
5. **Train classifier** (ResNet18)
6. **Evaluate** (accuracy/loss curves + confusion matrix)

---

## Input/Output Examples
Example: logo extracted → fake logo generated

![Example I/O](Screenshot%202026-01-29%20165351.png)

---

## Models and pipelines used

### 1) Synthetic data generation
- **StableDiffusionXLImg2ImgPipeline** (`stabilityai/stable-diffusion-xl-base-1.0`)
- Custom prompt templates per brand
- Optional typo forcing (probability controlled with `TYPO_PROB`)
- Post-processing defect functions (PIL/Numpy)

### 2) Classification model
- **ResNet18 (pretrained ImageNet)**  
- Final layer replaced for **2 classes**: `Real / Fake`

---

## Training process and parameters
**Data split:** 80% train / 20% validation (stratified)  
**Transforms (train):**
- Resize 256 → CenterCrop 224
- horizontal flip, rotation (±10°)
- color jitter, random grayscale
- normalize (ImageNet mean/std)

**Optimization:**
- Optimizer: `Adam(lr=1e-4)`
- Loss: `CrossEntropyLoss` + **class weights** (for class imbalance)
- Epochs: `15`
- Batch size: `16`

---

## Results
Training converged quickly with stable validation performance:

![Loss Curve](Screenshot%202026-01-29%20165338.png)
![Accuracy Curve](Screenshot%202026-01-29%20165332.png)

Confusion matrix (validation):

![Confusion Matrix](Screenshot%202026-01-29%20165346.png)

**Key takeaway:** After adding *low-quality Real samples*, the model stopped using image quality as a shortcut and focused more on *logo-specific features*.

---

## Repository structure
Suggested structure (lightweight + Git-friendly):

