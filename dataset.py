"""
Dual-Source Cyclone Dataset (IR + RAW) for FH-ODCNN
FINAL VERSION – 7 CLASS ONLY
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import skfuzzy as fuzz


class DualSourceCycloneDataset(Dataset):
    def __init__(self, csv_path, img_dir_ir, img_dir_raw, transform=None, preprocess=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir_ir = img_dir_ir
        self.img_dir_raw = img_dir_raw
        self.transform = transform
        self.preprocess_flag = preprocess

        self.samples = []

        for _, row in self.df.iterrows():
            img_name = row["img_name"]
            knots = float(row["label"])

            cls_idx = self._knots_to_class(knots)

            # Add IR image if exists
            ir_path = os.path.join(img_dir_ir, img_name)
            if os.path.exists(ir_path):
                self.samples.append((ir_path, "ir", cls_idx, knots))

            # Add RAW image if exists
            raw_path = os.path.join(img_dir_raw, img_name)
            if os.path.exists(raw_path):
                self.samples.append((raw_path, "raw", cls_idx, knots))

        # FINAL 7 CLASSES (IMD-ALIGNED)
        self.classes = [
            "LPA / No Cyclone",
            "Depression",
            "Deep Depression",
            "Cyclonic Storm",
            "Severe Cyclonic Storm",
            "Very Severe Cyclonic Storm",
            "Extremely Severe Cyclonic Storm",
        ]

    # -------------------------------------------------
    # 7-CLASS INTENSITY MAPPING (0–6)
    # -------------------------------------------------
    def _knots_to_class(self, knots):
        if knots < 28:
            return 0   # LPA / No Cyclone
        elif knots < 34:
            return 1   # Depression
        elif knots < 48:
            return 2   # Deep Depression
        elif knots < 64:
            return 3   # Cyclonic Storm
        elif knots < 90:
            return 4   # Severe Cyclonic Storm
        elif knots < 120:
            return 5   # Very Severe Cyclonic Storm
        else:
            return 6   # Extremely Severe Cyclonic Storm

    # -------------------------------------------------
    # FUZZY SEGMENTATION (INFERENCE / VISUALIZATION ONLY)
    # -------------------------------------------------
    def _fuzzy_segment(self, img_gray):
        img_small = cv2.resize(img_gray, (100, 100))
        flat = img_small.reshape(-1, 1).astype(np.float32) + 1e-5

        try:
            cntr, u, _, _, _, _, _ = fuzz.cmeans(
                flat.T, c=3, m=2, error=0.005, maxiter=100
            )
            cold_idx = np.argmin(cntr.flatten())
            membership = u[cold_idx, :].reshape(img_small.shape)
            membership = cv2.resize(
                membership, (img_gray.shape[1], img_gray.shape[0])
            )
            segmented = cv2.normalize(
                membership, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            return segmented
        except Exception:
            return img_gray

    # -------------------------------------------------
    # OPTIONAL PREPROCESSING (NOT USED DURING TRAINING)
    # -------------------------------------------------
    def _preprocess(self, img_path, source):
        if source == "ir":
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Cannot read IR image: {img_path}")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError(f"Cannot read RAW image: {img_path}")

        img_gray = cv2.resize(img_gray, (224, 224))
        bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        ahe = clahe.apply(bilateral)

        segmented = self._fuzzy_segment(ahe)
        blended = cv2.addWeighted(ahe, 0.6, segmented, 0.4, 0)

        return blended

    # -------------------------------------------------
    # STANDARD DATASET METHODS
    # -------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, source, cls_idx, knots = self.samples[idx]

        # SAFETY CHECK (CRITICAL)
        assert 0 <= cls_idx < 7, f"Invalid class index: {cls_idx}"

        if self.preprocess_flag:
            img = self._preprocess(img_path, source)
            img_pil = Image.fromarray(img)
        else:
            # Preserve IR color information
            if source == "ir":
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = cv2.resize(img, (224, 224))
            img_pil = Image.fromarray(img)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return (
            img_tensor,
            torch.tensor(cls_idx, dtype=torch.long),
            torch.tensor(knots, dtype=torch.float32),
        )
