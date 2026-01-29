# BrandGuard – GenAI-Based Pipeline for Counterfeit Sneaker Detection 👟🚫

**Automated detection of high-quality and low-quality counterfeit sneaker logos using synthetic data generation and Deep Learning.**

---

## 👥 Team Members
* **Tal Mitzmacher**
* **Amit Mitzmacher**
* **Danny Isserlis**

---

## 1. Project Motivation & Definition 🎯
**The Problem:** The counterfeit market is evolving. Distinguishing between a high-quality fake and an authentic product is becoming increasingly difficult due to a lack of documented fake data.

**Our Goal:** Develop a **Generative Data Pipeline** to create a synthetic dataset of counterfeits and train a classification model (**ResNet18**) to distinguish them with high precision.

**The Innovation:** Instead of relying on scarce real-world fake samples, our system generates variations that alter the logo's semantics (e.g., typos, graphic changes) without compromising its realistic texture.

---

## 2. Visual Abstract (Workflow) 🖼️

The system operates in a closed loop: extracting the authentic logo, generating sophisticated fakes using Stable Diffusion XL (SDXL), and training a classifier on the hybrid dataset.

```mermaid
graph TD
    subgraph Input
        A[Original Sneaker Image]
    end

    subgraph Preprocessing
        B[Logo Cropping]
        C[Data Balancing]
    end

    subgraph GenAI_Pipeline ["Generative AI Pipeline (SDXL)"]
        direction TB
        D{Dual-Strength Generation}
        D1[Low Strength 0.45-0.55<br/>Realistic/Subtle Fakes]
        D2[High Strength 0.70-0.88<br/>Aggressive/Typo Fakes]
        E[Quality Control<br/>Negative Prompts + Weighted Loss]
    end

    subgraph Classification
        F[ResNet18 Model<br/>Transfer Learning]
        G[Binary Classification<br/>Real vs Fake]
    end

    A --> B --> C
    C --> D
    D --> D1 & D2 --> E
    E --> F --> G
    
    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style D fill:#ffebee,stroke:#c62828,stroke-width:2px
    style F fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

```
