"""
Unified Validation Script (FINAL – 7 CLASS)
- Dataset structure & label sanity check
- Model inference evaluation
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from dataset import DualSourceCycloneDataset

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_PATH = "data/raw/labels.csv"
IMG_DIR_IR = "data/raw/ir_images"
IMG_DIR_RAW = "data/raw/raw_images"
MODEL_PATH = "model/fh_odcnn_best.pth"

BATCH_SIZE = 16
NUM_CLASSES = 7

CLASS_NAMES = [
    "LPA / No Cyclone",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm"
]

# --------------------------------------------------
# PART A: DATASET VALIDATION
# --------------------------------------------------
def validate_dataset():
    print("\n🔍 DATASET VALIDATION\n")

    if not os.path.exists(CSV_PATH):
        print("❌ labels.csv not found")
        return False

    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} labeled entries")
    print(f"   Columns: {list(df.columns)}\n")

    dataset = DualSourceCycloneDataset(
        csv_path=CSV_PATH,
        img_dir_ir=IMG_DIR_IR,
        img_dir_raw=IMG_DIR_RAW,
        transform=None,
        preprocess=False
    )

    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    for _, label, _ in dataset:
        class_counts[int(label)] += 1

    print(f"✅ Valid samples found: {len(dataset)}")
    print(f"   IR  : {sum(1 for s in dataset.samples if s[1] == 'ir')}")
    print(f"   RAW : {sum(1 for s in dataset.samples if s[1] == 'raw')}")

    print("\n📊 Class Distribution:")
    for i in range(NUM_CLASSES):
        print(f"   Class {i} ({CLASS_NAMES[i]}): {class_counts[i]}")

    # Dataset preview
    plt.figure(figsize=(15, 8))
    for i in range(min(6, len(dataset))):
        img_path, src, cls, knots = dataset.samples[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(
            f"{os.path.basename(img_path)}\n{knots} knots → {CLASS_NAMES[cls]}\n{src.upper()}",
            fontsize=9
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("dataset_preview.png", dpi=150)
    print("\n📸 Saved dataset preview → dataset_preview.png")

    return True


# --------------------------------------------------
# PART B: MODEL VALIDATION
# --------------------------------------------------
class FH_ODCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def validate_model():
    print("\n🧠 MODEL EVALUATION\n")

    model = FH_ODCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = DualSourceCycloneDataset(
        csv_path=CSV_PATH,
        img_dir_ir=IMG_DIR_IR,
        img_dir_raw=IMG_DIR_RAW,
        transform=transform,
        preprocess=False
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred, times = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            start = time.time()
            outputs = model(imgs)
            end = time.time()

            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            times.append(end - start)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    acc = np.trace(cm) / np.sum(cm)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    spec, fpr = [], []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec.append(TN / (TN + FP + 1e-8))
        fpr.append(FP / (FP + TN + 1e-8))

    print("📊 Performance Summary")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1-score        : {f1:.4f}")
    print(f"Specificity     : {np.mean(spec):.4f}")
    print(f"False Pos. Rate : {np.mean(fpr):.4f}")
    print(f"Avg time (ms)   : {(np.mean(times) * 1000):.2f}")

    print("\n📌 Confusion Matrix:")
    print(cm)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    ok = validate_dataset()
    if ok:
        validate_model()
    else:
        print("\n❌ Validation aborted due to dataset issues")
