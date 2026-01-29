# BrandGuard – GenAI-Based System for Counterfeit Sneaker Logo Detection 👟🚫

**Automated detection of high-quality and low-quality counterfeit sneaker logos using synthetic data generation and Deep Learning.**

---

## 1. Project Motivation 🎯
The global counterfeit sneaker market results in billions of dollars in losses. While spotting low-quality "knock-offs" is easy, AI-generated fakes are becoming indistinguishable. We created an automated AI system to classify sneaker logos with high precision.

---

## 2. Problem Statement ❓
Regular consumers lack the expertise to spot minute details in stitching or fonts.
* **Challenge:** Traditional models lack data on high-quality fakes.
* **Solution:** We generated a massive synthetic dataset using **Stable Diffusion XL** to train a robust **ResNet18** classifier.

---

## 3. Visual Abstract 🖼️
![Workflow Diagram](https://via.placeholder.com/800x200?text=Place+Your+Workflow+Image+Here)

---

## 4. Datasets Used 💾
We solved the data scarcity problem by creating a hybrid dataset:
* **Total Images:** 3,600
* **Real:** 1,800 (Augmented authentic logos).
* **Fake:** 1,800 (Synthetically generated).
* **Brands:** Nike, Adidas, Fila, New Balance, Converse, Jordan.

---

## 5. Data Generation Methods 🧬

### A. The GenAI Engine (High-Level Fakes - 40%)
Used **Stable Diffusion XL** with specific prompts:
* **Typos:** "Abibas", "Niky".
* **Geometry:** Distorted "Fatman" Jordan logo.

### B. The Distortion Engine (Low-Level Fakes - 60%)
Programmatic defects applied to images:
* `Blur` & `Noise` (Low quality camera simulation).
* `Swirl` & `Warp` (Physical deformities).

---

## 6. Repository Structure 📂

```text
BrandGuard/
├── data_generation/
│   ├── generate_fakes.py          # SDXL pipeline logic
│   └── post_processing.py         # Physical distortions
│
├── model_training/
│   ├── train_classifier.py        # ResNet18 training loop
│   └── prediction.py              # Inference script
│
├── results/                       # Evaluation graphs
│   ├── confusion_matrix.png
│   └── accuracy_loss_graph.png
│
└── README.md                      # Project documentation
