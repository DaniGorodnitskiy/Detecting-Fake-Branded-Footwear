BrandGuard – GenAI-Based System for Counterfeit Sneaker Logo Detection
Automated detection of high-quality and low-quality counterfeit sneaker logos using synthetic data generation and Deep Learning.

1. Project Motivation
The global counterfeit sneaker market results in billions of dollars in losses for major brands and consumers annually. While spotting low-quality "knock-offs" is relatively easy, the rise of high-quality manufacturing and Generative AI has made distinguishing between authentic and fake products increasingly difficult. We decided to create an automated, AI-driven system that can classify sneaker logos with high precision, specifically designed to handle the complexity of modern counterfeits.

2. Problem Statement
Standard consumers lack the expertise to identify minute details in stitching, logo geometry, or font typography that indicate a fake. Manual verification is slow, subjective, and expensive. Furthermore, traditional computer vision models often struggle because they lack a diverse dataset of "High-Level" fakes. This project aims to bridge that gap by generating a massive, hierarchically complex synthetic dataset to train a robust classifier capable of flagging even the most subtle discrepancies.

3. Visual Abstract
(Place an image here showing the flow: Original Image -> SDXL Generation -> Dataset -> ResNet18 -> Prediction)

4. Datasets Used or Collected
We faced a significant challenge: Data Scarcity. There is no publicly available, large-scale dataset of high-quality fake sneaker logos. To solve this, we collected a small seed dataset and expanded it synthetically:

Base Data: 30 high-resolution original images per brand.

Brands Covered: Nike, Adidas, Fila, New Balance, Converse, and Jordan.

Total Dataset Size: 3,600 Images.

Real Class (1,800 images): Augmented versions of authentic logos.

Fake Class (1,800 images): Synthetically generated counterfeits using GenAI.

5. Data Augmentation and Generation Methods
Synthetic Data Generation Pipeline
To create a robust classifier, we developed a sophisticated pipeline to generate valid "Fake" samples. The process is divided into two complexity levels:

1. The GenAI Engine (High-Level Fakes - 40%) We utilized Stable Diffusion XL (SDXL) via an Image-to-Image pipeline to reconstruct logos with semantic errors.

Prompt Engineering: We crafted specific prompts to simulate common counterfeit traits, such as:

Typos: "Abibas" (Adidas), "Niky" (Nike), "Cornverse" (Converse).

Geometry: "Fatman" silhouette (Jordan), missing stars, disconnected letters.

Texture: Wrong leather grain, plastic-looking heel tabs.

Strength Control: We varied the strength parameter (0.70 - 0.88) to allow the AI to hallucinate realistic but incorrect details.

2. The Distortion Engine (Low-Level Fakes - 60%) We applied programmatic transformations to simulate poor manufacturing quality:

Physical Distortion: Using skimage.transform.swirl to physically warp the logo geometry.

Quality Degradation: Gaussian blurring, noise injection, and downscaling to simulate low-resolution prints.

Structural Defects: Thinning/thickening lines and edge jittering.

3. Authentic Augmentation To balance the dataset, the "Real" images underwent strict geometric augmentations (rotation, lighting changes) that preserved the integrity of the logo while adding variance.

6. Input/Output Examples
(Place an image here showing a side-by-side comparison: A Real Nike Logo vs. a Generated "Niky" Fake)

7. Models and Pipelines Used
Data Generation Pipeline
Model: Stable Diffusion XL Base 1.0 (stabilityai/stable-diffusion-xl-base-1.0).

Library: Hugging Face diffusers.

Pipeline: StableDiffusionXLImg2ImgPipeline. This allowed us to guide the generation using the original image geometry while injecting new (fake) semantic information via text prompts.

Classification Pipeline
The core system for detecting discrepancies operates on a fine-tuned Convolutional Neural Network (CNN).

Architecture: ResNet18 (Pre-trained on ImageNet).

Modification: The fully connected layer (fc) was modified to output 2 classes: Real vs Fake.

Loss Function: CrossEntropyLoss weighted to handle any potential class imbalance.

Optimization: The model was trained using the Adam optimizer with a learning rate of 1e-4.

8. Training Process and Parameters
The binary classification model was trained on the hybrid (Real + Synthetic) dataset.

Dataset Split: 80% Training / 20% Validation (Stratified to ensure brand balance).

Batch Size: 16.

Epochs: 15 (with Early Stopping logic via model checkpointing).

Input Resolution: 224x224 (Center Cropped).

Normalization: Standard ImageNet mean/std [0.485, 0.456, 0.406].

Hierarchy Strategy: We intentionally designed the dataset split to be 60% Low-Level and 40% High-Level fakes. This curriculum learning approach ensures the model learns basic defect detection (blur/noise) while also mastering the identification of subtle, photorealistic anomalies (GenAI fakes).

9. Metrics
To evaluate performance, we focused on metrics that prioritize the detection of fakes (minimizing False Negatives).

Accuracy: The overall correctness of the model.

Recall (Sensitivity): Crucial for this domain. A False Negative (calling a fake shoe "Real") is the worst-case scenario. High recall ensures we catch the fakes.

Confusion Matrix: Used to visualize the specific types of errors the model makes.

Loss Curves: We monitored Training vs. Validation loss to ensure the model was generalizing and not just memorizing the synthetic images (Overfitting).

10. Results
We evaluated the pipeline on a held-out validation set of 720 images.

Classification Performance: The ResNet18 model demonstrated exceptional ability to distinguish between authentic and AI-generated fakes.

Accuracy: ~97%.

Generalization: The Training and Validation accuracy curves converged closely, indicating no overfitting.

Confusion Matrix Analysis:

True Positives (Correctly identified Fakes): 348

True Negatives (Correctly identified Real): 350

False Negatives (Missed Fakes): Only 12

False Positives (False Alarms): Only 10

Qualitative Assessment: The model successfully flagged both obvious distortions (swirled logos) and subtle semantic changes (e.g., "Abibas" text or a slightly malformed Jordan Jumpman), proving the effectiveness of the GenAI training data.

11. Repository Structure
Plaintext
BrandGuard/
├── data_generation/
│   ├── generate_fakes.py          # SDXL pipeline for synthetic data generation
│   └── post_processing.py         # Physical distortion logic (swirl, blur)
│
├── model_training/
│   ├── train_classifier.py        # ResNet18 training loop and evaluation
│   └── prediction.py              # Inference script for testing new images
│
├── notebooks/
│   ├── Data_Gen_SDXL.ipynb        # Colab notebook for data generation
│   └── Model_Training.ipynb       # Colab notebook for training
│
├── data/                          # (Not included in repo due to size)
│   ├── original_logo/             # Source images
│   └── dataset_v1/                # The generated dataset
│
├── results/                       # Evaluation graphs and matrices
│   ├── confusion_matrix.png
│   └── accuracy_loss_graph.png
│
├── requirements.txt               # List of Python dependencies
└── README.md                      # Project documentation
12. Team Members
[Name 1]

[Name 2]

[Name 3]

13. Installation Guide
Required Models This project automatically downloads the stable-diffusion-xl-base-1.0 model and resnet18 weights from Hugging Face and PyTorch Hub respectively.

Setup

Clone the repository:

Bash
git clone https://github.com/your-username/BrandGuard.git
cd BrandGuard
Install dependencies:

Bash
pip install -r requirements.txt
Run Data Generation:

Bash
python data_generation/generate_fakes.py
Run Training:

Bash
python model_training/train_classifier.py
