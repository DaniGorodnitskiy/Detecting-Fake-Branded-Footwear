# =========================
# IMPORT
# =========================

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from google.colab import drive
from PIL import Image
import matplotlib.pyplot as plt

# חיבור ל-Google Drive
drive.mount('/content/drive')

# בסיס תיקיות בדרייב
BASE_DIR = "/content/drive/MyDrive/originals VS fake"

# כאן נמצאים כל התמונות האמיתיות (לוגו תקין + וריאציות סינטטיות)
ORIGINALS_ROOT = os.path.join(BASE_DIR, "original_logo")

# כאן נמצאים כל הזיופים (לוגו משובש על נעל)
FAKES_ROOT = os.path.join(BASE_DIR, "counterfeit")

# תיקיית עבודה לאימון (נבנה בה Real/Fake)
TRAIN_DATASET_DIR = "/content/training_data"

# איפה נשמור את המודל המאומן
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "logo_resnet18.pth")

# התקן (GPU אם יש)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# טרנספורמציות – איך מעבדים את התמונות לפני המודל
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),             # מתקרב למרכז (איפה שהלוגו בערך)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
}


# =========================
# DATA SCAN
# =========================

def prepare_dataset():
    """
    בונה מבנה:
      /content/training_data/
          Real/
          Fake/
    ומעתיק לשם את כל התמונות מ:
      ORIGINALS_ROOT/*/*  -> Real (authentic)
      FAKES_ROOT/*/*      -> Fake (counterfeit)
    """
    if os.path.exists(TRAIN_DATASET_DIR):
        shutil.rmtree(TRAIN_DATASET_DIR)

    real_dir = os.path.join(TRAIN_DATASET_DIR, "Real")
    fake_dir = os.path.join(TRAIN_DATASET_DIR, "Fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Real = original_logo (לוגו אמיתי על הנעל, כולל וריאציות)
    print("Collecting Real images...")
    real_count = 0
    for root, _, files in os.walk(ORIGINALS_ROOT):
        for img_name in files:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(root, img_name)
                dst = os.path.join(real_dir, f"r_{real_count}_{img_name}")
                shutil.copy(src, dst)
                real_count += 1

    # Fake = counterfeit (לוגו מזויף על נעל)
    print("Collecting Fake images...")
    fake_count = 0
    for root, _, files in os.walk(FAKES_ROOT):
        for img_name in files:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(root, img_name)
                dst = os.path.join(fake_dir, f"f_{fake_count}_{img_name}")
                shutil.copy(src, dst)
                fake_count += 1

    print("\n--- SUCCESS ---")
    print(f"Real: {real_count} | Fake: {fake_count}")
    return real_count, fake_count

real_count, fake_count = prepare_dataset()

if real_count == 0 or fake_count == 0:
    raise ValueError("⚠️ אחת מהתיקיות ריקה! בדוק את ORIGINALS_ROOT / FAKES_ROOT והתמונות בתוכן.")
else:
    print("Dataset looks OK, continuing...")


# =========================
# STUDY MODEL - old
# =========================

# 1. טוענים את כל הדאטה כדי לקבל targets
base_dataset = datasets.ImageFolder(TRAIN_DATASET_DIR)
targets = base_dataset.targets
class_names = base_dataset.classes   # ['Fake', 'Real'] לפי ABC
print("Classes:", class_names)

# 2. חלוקה ל-Train / Validation
train_idx, val_idx = train_test_split(
    np.arange(len(base_dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset_full = datasets.ImageFolder(TRAIN_DATASET_DIR, transform=data_transforms['train'])
val_dataset_full   = datasets.ImageFolder(TRAIN_DATASET_DIR, transform=data_transforms['val'])

train_dataset = Subset(train_dataset_full, train_idx)
val_dataset   = Subset(val_dataset_full,   val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples  : {len(val_dataset)}")

# 3. מודל ResNet18 פרה-טריינד
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 מחלקות: Fake / Real
model = model.to(DEVICE)

# 4. חישוב משקלים למחלקות (אם יש חוסר איזון)
targets_np = np.array(targets)
class_counts = np.bincount(targets_np)  # [cnt_Fake, cnt_Real]
num_classes = len(class_counts)
total_samples = len(targets_np)

class_weights = total_samples / (num_classes * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print("Class counts :", class_counts)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 5. לולאת אימון

def train_model(epochs=15):
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        # === TRAIN ===
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc  = running_corrects.double() / len(train_dataset)

        # === VAL ===
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc  = val_corrects.double() / len(val_dataset)

        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {val_epoch_loss:.4f} | Val   Acc: {val_epoch_acc:.4f}")

        # שומרים את המודל הכי טוב לפי דיוק ולידציה
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_state = model.state_dict()

    if best_state is not None:
        torch.save(best_state, MODEL_PATH)
        print(f"\nBest model saved to: {MODEL_PATH}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
    else:
        print("\n⚠️ WARNING: no best model state was saved!")

# להריץ אימון:
train_model(epochs=15)


# =========================
# STUDY MODEL (UPDATED)
# =========================

# 1. טוענים את כל הדאטה כדי לקבל targets
base_dataset = datasets.ImageFolder(TRAIN_DATASET_DIR)
targets = base_dataset.targets
class_names = base_dataset.classes   # ['Fake', 'Real']
print("Classes:", class_names)

# 2. חלוקה ל-Train / Validation
train_idx, val_idx = train_test_split(
    np.arange(len(base_dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset_full = datasets.ImageFolder(TRAIN_DATASET_DIR, transform=data_transforms['train'])
val_dataset_full   = datasets.ImageFolder(TRAIN_DATASET_DIR, transform=data_transforms['val'])

train_dataset = Subset(train_dataset_full, train_idx)
val_dataset   = Subset(val_dataset_full,   val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples  : {len(val_dataset)}")

# 3. מודל ResNet18 פרה-טריינד
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# 4. חישוב משקלים למחלקות
targets_np = np.array(targets)
class_counts = np.bincount(targets_np)
num_classes = len(class_counts)
total_samples = len(targets_np)

class_weights = total_samples / (num_classes * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 5. לולאת אימון מעודכנת לשמירת היסטוריה
def train_model(epochs=15):
    # מילון לשמירת הנתונים עבור הגרפים
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        # === TRAIN ===
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc  = running_corrects.double() / len(train_dataset)

        # === VAL ===
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc  = val_corrects.double() / len(val_dataset)

        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {val_epoch_loss:.4f} | Val   Acc: {val_epoch_acc:.4f}")

        # שמירת נתונים להיסטוריה
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_state = model.state_dict()

    if best_state is not None:
        torch.save(best_state, MODEL_PATH)
        print(f"\nBest model saved. Best Val Accuracy: {best_val_acc:.4f}")

    return history # מחזיר את הנתונים לגרפים

# הרצת האימון ושמירת התוצאות
history = train_model(epochs=15)

# =========================
# CHECK THE PICTURE
# =========================

def check_logo(image_path):
    """
    מקבלת path לתמונה (עדיף crop של אזור הלוגו על הנעל),
    מציגה את התמונה + מנבאת: Fake / Real + אחוז ביטחון.
    """
    if not os.path.exists(MODEL_PATH):
        print("Model file not found:", MODEL_PATH)
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img_t = data_transforms['val'](img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_class = class_names[pred.item()]   # 'Fake' או 'Real'
    confidence = conf.item() * 100

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# דוגמה לשימוש:
check_logo("/content/drive/MyDrive/originals VS fake/testing/fakeFila1.png")


import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. גרף לימוד (Loss & Accuracy) ---
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_history(history)

# --- 2. מטריצת בלבול וטבלת מדדים ---
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # מטריצת בלבול
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # טבלת מדדים (Precision, Recall, F1)
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

evaluate_model()