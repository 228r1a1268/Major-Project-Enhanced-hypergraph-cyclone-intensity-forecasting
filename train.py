"""
FH-ODCNN Training Script (FINAL – Stratified Split Version)
7-class cyclone intensity classification
ResNet-18 backbone
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from dataset import DualSourceCycloneDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting training on {DEVICE}\n")

CSV_PATH = "data/raw/labels.csv"
IMG_DIR_IR = "data/raw/ir_images"
IMG_DIR_RAW = "data/raw/raw_images"

NUM_CLASSES = 7
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
SEED = 42

torch.manual_seed(SEED)

# =====================
# TRANSFORMS
# =====================
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# DATASET
# =====================
print("Loading INSAT-3D dataset...")

full_dataset = DualSourceCycloneDataset(
    csv_path=CSV_PATH,
    img_dir_ir=IMG_DIR_IR,
    img_dir_raw=IMG_DIR_RAW,
    transform=None,
    preprocess=False
)

# =====================
# STRATIFIED SPLIT
# =====================
labels_list = []
for _, label, _ in full_dataset:
    labels_list.append(label.item())

labels_array = np.array(labels_list)

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

for train_idx, val_idx in sss.split(np.zeros(len(labels_array)), labels_array):
    train_indices = train_idx
    val_indices = val_idx

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Assign transforms
train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total samples: {len(full_dataset)}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
print(f"Classes: {NUM_CLASSES}\n")

# =====================
# CLASS DISTRIBUTION
# =====================
class_counts = [0] * NUM_CLASSES
for _, label, _ in full_dataset:
    class_counts[label.item()] += 1

print("Class Distribution:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count} samples")

class_counts = np.array(class_counts, dtype=np.float32)
class_weights = 1.0 / (np.sqrt(class_counts) + 1e-5)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print(f"\nClass weights (sqrt-based): {class_weights.cpu().numpy()}\n")

# =====================
# MODEL
# =====================
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
        features = self.backbone(x)
        return self.classifier(features)

model = FH_ODCNN().to(DEVICE)

# =====================
# OPTIMIZATION
# =====================
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

# =====================
# TRAIN / VALIDATE
# =====================
def train_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels, _ in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# =====================
# TRAIN LOOP
# =====================
best_acc = 0.0
train_accs, val_accs = [], []

print("Starting training...\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/fh_odcnn_best.pth")
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} ★ (Best)")
    else:
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    if epoch > 20 and val_acc < best_acc * 0.95:
        print("\nEarly stopping triggered")
        break

print(f"\nTraining complete! Best validation accuracy: {best_acc:.2%}")

# Save training curve
plt.figure(figsize=(10, 4))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Training curves saved.")