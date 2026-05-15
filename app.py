"""
FH-ODCNN Web Application 
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import skfuzzy as fuzz
from flask import Flask, render_template, request, send_from_directory

# ----------------------------------------
# Flask Setup
# ----------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["PRED_FOLDER"] = "static/predictions"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PRED_FOLDER"], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting FH-ODCNN web app on {DEVICE}")

# ----------------------------------------
# Model Definition (7 Classes)
# ----------------------------------------

class FH_ODCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


model = FH_ODCNN(num_classes=7).to(DEVICE)
MODEL_PATH = "model/fh_odcnn_best.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded successfully.")
else:
    raise FileNotFoundError("Model file not found. Train the model first.")

model.eval()

# ----------------------------------------
# IMD 7-Class Labels
# ----------------------------------------

CLASS_NAMES = [
    "LPA / No Cyclone",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm"
]

KNOTS_RANGES = [
    "< 28 knots",
    "28-33 knots",
    "34-47 knots",
    "48-63 knots",
    "64-89 knots",
    "90-119 knots",
    "≥ 120 knots"
]

CLASS_DESCRIPTIONS = [
    "Stable low-pressure system with no organized cyclone structure.",
    "Weak cyclonic circulation with limited wind intensification.",
    "Strengthening system with sustained convective activity.",
    "Organized cyclone with moderate sustained winds.",
    "Strong cyclone posing regional impact risk.",
    "High-intensity cyclone with significant wind damage potential.",
    "Extreme cyclone with severe destructive capacity."
]

# ----------------------------------------
# Preprocessing + Fuzzy Segmentation
# ----------------------------------------

def preprocess_fh_odcnn(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    ahe = clahe.apply(bilateral)

    img_small = cv2.resize(ahe, (100, 100))
    flat = img_small.reshape(-1, 1).astype(np.float32) + 1e-5

    try:
        cntr, u, _, _, _, _, _ = fuzz.cmeans(flat.T, 3, 2, error=0.005, maxiter=100)
        cold_idx = np.argmin(cntr.flatten())
        membership = u[cold_idx, :].reshape(img_small.shape)
        membership = cv2.resize(membership, (224, 224))
        segmented = cv2.normalize(
            membership, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    except:
        segmented = ahe

    cv2.imwrite(os.path.join(app.config["PRED_FOLDER"], "bilateral.png"), bilateral)
    cv2.imwrite(os.path.join(app.config["PRED_FOLDER"], "ahe.png"), ahe)
    cv2.imwrite(os.path.join(app.config["PRED_FOLDER"], "segmented.png"), segmented)
    cv2.imwrite(
        os.path.join(app.config["PRED_FOLDER"], "blended_input.png"),
        cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    )

    return img_resized

# ----------------------------------------
# Transform
# ----------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------
# Routes
# ----------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/forecast", methods=["POST"])
def forecast():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        processed_img = preprocess_fh_odcnn(filepath)
        img_pil = Image.fromarray(processed_img)
        tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item() * 100

        # ----------------------------------------
        # Overlay Final Prediction Image
        # ----------------------------------------

        overlay = processed_img.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (overlay.shape[1] - 10, 110),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            overlay,
            "FH-ODCNN Prediction",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            f"{CLASS_NAMES[pred_idx]} ({KNOTS_RANGES[pred_idx]})",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            f"Confidence: {confidence:.1f}%",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imwrite(
            os.path.join(app.config["PRED_FOLDER"], "overlay_forecast.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        result = {
            "headline": f"{CLASS_NAMES[pred_idx]} - {CLASS_DESCRIPTIONS[pred_idx]}",
            "knots_range": KNOTS_RANGES[pred_idx],
            "confidence": f"{confidence:.1f}%"
}

        images = [
            "blended_input.png",
            "bilateral.png",
            "ahe.png",
            "segmented.png",
            "overlay_forecast.png"
        ]

        return render_template(
            "index.html",
            filename=file.filename,
            result=result,
            images=images
        )

    except Exception as e:
        return f"Error processing image: {str(e)}", 500

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)