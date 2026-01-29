# Counterfeit Logo Detection (GenAI + CV)

> **TL;DR:** We built an end-to-end pipeline that **generates synthetic counterfeit logo images** using **Stable Diffusion XL (img2img)** and **post-processing defects**, then **trains a ResNet18 classifier** to predict **Real vs Fake** from logo images—even under varied backgrounds and lighting.

---

## Project Motivation
Counterfeit products are a real-world problem in fashion and retail. In practice, **high-quality labeled counterfeit datasets are scarce** (and inconsistent), which makes it hard to train reliable models.

Our project tackles this by combining **Generative AI (diffusion models)** with **classic computer vision training**:
- Use a small set of authentic logos as seeds
- Generate many realistic and adversarial counterfeit variations
- Train a classifier that learns the *visual concept of authenticity* rather than “cheap photo = fake”

---

## Problem Statement
Given an input image containing a brand logo (on fabric, rubber, embroidery, etc.), classify it as:

- **Real** — authentic logo appearance
- **Fake** — counterfeit indicators such as:
  - altered typography (e.g., `NIKE → NYKE`)
  - distorted strokes and thickness
  - misaligned edges and outlines
  - perspective warping, blur, double-print, missing parts

**Goal:** build a model that is robust to background changes and photography conditions, and focuses on logo authenticity.

---

## Visual Abstract
1. **Collect** a small set of authentic logo images (per brand)  
2. **Generate** realistic variants of authentic logos (brightness/rotation + low quality)  
3. **Generate** counterfeit logos using SDXL img2img  
4. **Apply** additional counterfeit defects (edge jitter, perspective warp, missing letter, etc.)  
5. **Train** a CNN (ResNet18) to classify **Real vs Fake**  
6. **Evaluate** via learning curves + confusion matrix + classification report  

---

## Datasets Used or Collected
### 1) Real Logos (Collected)
- Stored under:
  - `original_logo/<BRAND_NAME>/`
- These are the seed images used for both:
  - authentic augmentation (Real class)
  - diffusion-based counterfeit generation (Fake class)

### 2) Synthetic Authentic Variants (Generated)
Stored under:
- `real_augmented/<BRAND_NAME>/`

Generated via:
- small rotations
- brightness shifts
- intentionally low-quality copies (to prevent learning “low quality = fake” shortcut)

### 3) Synthetic Counterfeits (Generated)
Stored under:
- `counterfeit/<BRAND_NAME>/`

Generated via:
- SDXL img2img prompting
- plus post-processing defects

---

## Data Augmentation and Generation Methods

### A) Authentic augmentation (Real class)
Implemented in `generate_authentic_variations()`:
- `Random rotation (-5° to +5°)`
- `Brightness jitter (0.95–1.05)`
- **Low-quality simulation**:
  - downscale → upscale
  - blur
  - gaussian noise

> Motivation: We discovered the classifier tends to learn **artifact shortcuts** (“low quality = fake”).  
> Fix: Add low-quality authentic examples to the Real class.

---

### B) Diffusion generation (Fake class) — SDXL Img2Img
Implemented in `generate_ai_fakes()` with:
- `StableDiffusionXLImg2ImgPipeline`
- Two modes:
  - **LOW strength** (~0.45–0.55): subtle counterfeits
  - **HIGH strength** (~0.70–0.88): aggressive counterfeits + misspellings

---

### C) Post-generation counterfeit defects (POST_DEFECTS)
After SDXL generation, we apply one random “defect” to simulate real counterfeit signs:

| Defect | What it simulates |
|---|---|
| `thin` / `thicken` | wrong logo stroke thickness |
| `missing` | missing patch / incomplete print |
| `doubleprint` | double stamping / offset print |
| `coloroff` | wrong colors / saturation |
| `blur` | poor print sharpness |
| `perspective` | camera angle distortion |
| `outline` | unnatural thick outline |
| `edgejitter` | shaky edges / rough stitching |
| `missingletter` | cropped/erased character |

**Implementation detail:**  
We create a rough logo mask by assuming **non-white pixels** belong to the logo, then apply morphological operations and alpha compositing.

---

### D) Typo / misspelling injection (TYPO_VARIANTS)
Controlled by:
- `TYPO_PROB = 0.3`

Example:
- `Nike → NYKE`, `NIKY`, `N1KE`, `NIIE`
- `Converse → CONVESE`, `CORNVERSE`
- etc.

This forces the generator to produce semantic counterfeit signals (not only blur/noise).

---

## Workflow Visualization
(1) Seed Real Logos
original_logo/<brand>/*.png
|
v
(2) Real Augmentation
rotation + brightness + low quality
-> real_augmented/<brand>/
|
+-----------------------------+
| |
v v
(3) SDXL Img2Img Fake Generation (4) Dataset Build
subtle + aggressive strengths training_data/Real, Fake
+ typo injection
+ post-defects
-> counterfeit/<brand>/
|
v
(5) Train ResNet18 Classifier
PyTorch + ImageFolder + transforms
|
v
(6) Evaluate + Predict
curves + confusion matrix + single-image inference


---

## Input/Output Examples
### Input
- A logo image (on shoe tongue / patch / embroidery), any background.

### Output
- `Real` or `Fake` with confidence score

Example inference call:
```python
check_logo("/content/drive/MyDrive/originals VS fake/testing/fakeFila1.png")
Models and Pipelines Used
Generative Model
StableDiffusionXLImg2ImgPipeline

Model checkpoint:

stabilityai/stable-diffusion-xl-base-1.0

Classifier
ResNet18 (pretrained)

Final layer replaced with 2-class head:

Fake / Real

Training Process and Parameters
Dataset building
Creates:

/content/training_data/
  Real/
  Fake/
By copying:

original_logo/** → Real

counterfeit/** → Fake

Preprocessing / Augmentations (Classifier)
Train transforms:

Resize(256) + CenterCrop(224)

Horizontal flip

Rotation(10)

Color jitter

Random grayscale

Normalize (ImageNet)

Validation transforms:

Resize(256) + CenterCrop(224)

Normalize (ImageNet)

Training config
Optimizer: Adam

LR: 1e-4

Loss: CrossEntropyLoss with class weights (handles imbalance)

Batch size: 16

Epochs: 15

Best checkpoint: saved to:

models/logo_resnet18.pth

Results
We report:

Train/Validation Accuracy

Train/Validation Loss

Confusion Matrix

Precision / Recall / F1 via classification report

Note: final performance depends on dataset size, brand variety, and how realistic the synthetic defects are.

Repository Structure
Suggested structure for the repo:

.
├── notebooks/
│   └── main_pipeline.ipynb
├── src/
│   ├── generation/
│   │   ├── sdxl_fake_generator.py
│   │   └── post_defects.py
│   ├── training/
│   │   ├── dataset_builder.py
│   │   ├── train_resnet.py
│   │   └── evaluate.py
│   └── inference/
│       └── predict.py
├── assets/
│   ├── visual_abstract.png
│   ├── examples_real_fake.png
│   └── confusion_matrix.png
├── requirements.txt
├── README.md
└── LICENSE
Team Members
Daniel (you)

<Member 2>

<Member 3>

How to Run (Colab-friendly)
Mount Drive:

from google.colab import drive
drive.mount('/content/drive')
Generate data:

set BRAND_NAME

ensure seed logos exist in original_logo/<brand>/

run generator to populate:

real_augmented/<brand>/

counterfeit/<brand>/

Prepare dataset:

runs prepare_dataset() → creates /content/training_data

Train:

run train_model(epochs=15)

Evaluate:

plot_history(history)

evaluate_model()

Inference:

check_logo(path_to_image)

Notes / Lessons Learned
Shortcut learning: Model initially learned that “low quality = fake”.
✅ Fixed by adding low-quality authentic images into the Real class.

Background diversity: Since your backgrounds are highly varied, heavy augmentation is critical.
✅ Used color jitter + grayscale + rotation + normalization.

Synthetic data realism: Post-defects improve realism beyond pure diffusion output.
