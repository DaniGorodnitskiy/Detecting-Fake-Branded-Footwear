# BrandGuard – GenAI-Based System for Counterfeit Sneaker Logo Detection 👟

**Automated detection of high-quality and low-quality counterfeit sneaker logos using synthetic data generation and Deep Learning.**

---

## 1. Project Motivation 
The global counterfeit sneaker market results in billions of dollars in losses. While spotting low-quality "knock-offs" is easy, AI-generated fakes are becoming indistinguishable. We created an automated AI system to classify sneaker logos with high precision.

---

## 2. Problem Statement 
Regular consumers lack the expertise to spot minute details in stitching or fonts.
* **Challenge:** Traditional models lack data on high-quality fakes.
* **Solution:** We generated a massive synthetic dataset using **Stable Diffusion XL** to train a robust **ResNet18** classifier.

---

## 3. Visual Abstract 
graph TD
    subgraph Data_Collection ["1. Data Collection"]
        A[Original Images<br/>(30 per Brand)]
    end

    subgraph Synthetic_Generation ["2. Synthetic Data Generation"]
        direction TB
        B[Augmentation Pipeline<br/>Rotate, Light, Scale]
        C[GenAI Pipeline<br/>Stable Diffusion XL]
        
        C1[Low-Level Fakes 60%<br/>Blur, Noise, Distortion]
        C2[High-Level Fakes 40%<br/>Typos, AI Hallucinations]
        
        C --> C1
        C --> C2
    end

    subgraph Dataset_Creation ["3. Dataset Compilation"]
        D{Hybrid Dataset<br/>3,600 Images}
        D1[Real Class<br/>1,800 Images]
        D2[Fake Class<br/>1,800 Images]
        
        A --> B --> D1 --> D
        A --> C --> D2 --> D
    end

    subgraph Model_Training ["4. Model Training"]
        E[ResNet18 CNN<br/>Pre-trained on ImageNet]
        F[Training Loop<br/>Epochs: 15, Batch: 16]
        
        D --> E --> F
    end

    subgraph Evaluation ["5. Evaluation"]
        G[Confusion Matrix]
        H[Accuracy & Loss Curves]
        I[Final Prediction<br/>Real vs Fake]
        
        F --> G
        F --> H
        F --> I
    end

    style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style C fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    style D fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style E fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style I fill:#f3e5f5,stroke:#4a148c,stroke-width:2px

---

## 4. Datasets Used 
We solved the data scarcity problem by creating a hybrid dataset:
* **Total Images:** 3,600
* **Real:** 1,800 (Augmented authentic logos).
* **Fake:** 1,800 (Synthetically generated).
* **Brands:** Nike, Adidas, Fila, New Balance, Converse, Jordan.

---

## 5. Data Generation Methods 

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

## 👥 Team Members

This project was built by a dedicated team of Data Science students.

| Name | Role & Responsibilities | GitHub / LinkedIn |
| :--- | :--- | :--- |
| **[שם חבר 1]** | **Lead Data Engineer**<br>Responsible for the GenAI pipeline, Prompt Engineering (SDXL), and synthetic data generation logic. | [@User1](https://github.com/) |
| **[שם חבר 2]** | **Model Architect**<br>In charge of the CNN architecture (ResNet18), training optimization, and hyperparameter tuning. | [@User2](https://github.com/) |
| **[שם חבר 3]** | **Data Analyst & Researcher**<br>Managed data collection, dataset balancing strategy, and performance evaluation metrics. | [@User3](https://github.com/) |

## 🚧 Challenges & Solutions

During the development, we encountered several key challenges:

### 1. The "Perfect Fake" Paradox
* **Challenge:** Initially, Stable Diffusion generated logos that were *too perfect*, making them indistinguishable from real ones even for humans.
* **Solution:** We implemented a **"Defect Injection"** logic. We adjusted the `strength` parameter (0.70-0.88) and used specific prompts like *"messy stitching"* and *"misspelled text"* to force the AI to create detectable flaws.

### 2. Data Leakage
* **Challenge:** Ensuring that variations of the same original image don't end up in both the Training and Validation sets.
* **Solution:** We implemented a strict split based on the base image ID, ensuring complete separation between the data the model learns from and the data it is tested on.

### 3. Class Imbalance
* **Challenge:** Real images were scarce compared to the infinite potential of generating fakes.
* **Solution:** We balanced the dataset 50/50 (1,800 Real / 1,800 Fake) by using aggressive geometric augmentation on the Real class to match the volume of the Synthetic Fake class.
