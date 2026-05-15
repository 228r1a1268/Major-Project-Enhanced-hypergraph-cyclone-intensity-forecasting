FH-ODCNN : Enhanced Hypergraph Cyclone Intensity Forecasting
-------------------------------------------------------------------------------

**ABOUT THE PROJECT:**
-----------------------------------------------------
FH-ODCNN (Fuzzy Hypergraph Optimized Deep Convolutional Neural Network) is a deep learning system for classifying cyclone intensity from satellite imagery. It uses a ResNet-18 backbone combined with Fuzzy C-Means segmentation and CLAHE-based image enhancement to classify cyclones into 7 intensity categories aligned with IMD (India Meteorological Department) standards. The model is trained on dual-source INSAT-3D satellite data (IR and RAW channels) with stratified class balancing and achieves strong performance across all intensity classes. The pipeline processes raw satellite images through bilateral filtering, adaptive histogram equalization, and fuzzy segmentation before classification  making it robust to noise and varying image conditions.

🌀 Live Demo:
--------------------------------------------------------------------------
https://major-project-enhanced-hypergraph-cyclone-intens-production.up.railway.app

Note: The app may take 20-30 seconds to load if it has been idle. This is normal just wait for it to wake up.

**HOW TO USE THE APP:**
--------------------------------------------------------------


**STEP-1 :** Get a Cyclone Satellite Image.
-----------------------------------------------------------
You need a satellite image of a cyclone or weather system. Good sources:

INSAT-3D / MOSDAC : Indian satellite imagery
NASA Worldview : Global satellite data
Any IR (infrared) or visible satellite image of a tropical system
Supported formats: JPG, PNG, JPEG

**STEP-2 :** Upload the Image.
----------------------------------------------------
On the app homepage, click the "Choose File" or upload button and select your satellite image from your device.

**NOTE:** There is no alert or notification to let you know the image is uploaded or not, for time being you can check by directly clicking on run button.

**STEP-3:** Run Prediction.
---------------------------------------------------
Click the "Run Prediction" or "Forecast" button. The system will process your image through the full FH-ODCNN pipeline.

**STEP-4:** Read the Output
---------------------------------------------------
The app returns:
Output FieldWhat It MeansClass NameThe IMD intensity category (e.g. "Cyclonic Storm")Wind Speed Range_Estimated sustained wind speed in knotsConfidenceHow confident the model is in its prediction (%)Processing Images5 visualizations showing each step of the pipeline.

**Understanding the Output**
-------------------------------------------------------------------------------------------------
The 7 Intensity Classes:

| Class | Name | Wind Speed | Meaning |
|---|---|---|---|
| 0 | LPA / No Cyclone | < 28 knots | Stable low-pressure, no cyclone structure |
| 1 | Depression | 28–33 knots | Weak cyclonic circulation forming |
| 2 | Deep Depression | 34–47 knots | Strengthening system with organized convection |
| 3 | Cyclonic Storm | 48–63 knots | Organized cyclone with moderate winds |
| 4 | Severe Cyclonic Storm | 64–89 knots | Strong cyclone, regional impact risk |
| 5 | Very Severe Cyclonic Storm | 90–119 knots | High-intensity, significant wind damage |
| 6 | Extremely Severe Cyclonic Storm | ≥ 120 knots | Extreme cyclone, severe destructive capacity |

The 5 Processing Images Explained:
The app shows you exactly what the model "sees" at each stage:
-----------------------------------------------------------------------
Blended Input : Your original image resized to 224×224 pixels.

Bilateral Filter : Noise removed while preserving edges (cloud boundaries).

AHE (Adaptive Histogram Equalization) : Contrast enhanced to reveal cyclone structure.

Segmented : Fuzzy C-Means segmentation highlighting the cold cloud tops (the most intense part of the cyclone).

Overlay Forecast : Final prediction overlaid on the image with class label, wind range, and confidence score.

Confidence Score:
----------------------------------------------------
Above 80% : High confidence, reliable prediction.

60–80% : Moderate confidence, prediction is likely correct.

Below 60% : Low confidence, image may be unclear or ambiguous.

What Kind of Images Work Best:
------------------------------------------
✅ Good inputs:
-------------------------------------------------
IR (infrared) satellite images showing cloud top temperatures.

Clear cyclone structure visible (eye, eyewall, spiral bands).

INSAT-3D, GOES, or Himawari satellite imagery.


❌ Poor inputs:
-----------------------------------------
Regular photographs or news images.

Very low resolution images.

Images with heavy text/watermarks covering the cyclone.

Visible-light images with heavy cloud cover masking structure.


**TECH STACK:**
--------------------------------------------------
Model: ResNet-18 + custom classifier head.

Preprocessing: Bilateral filter → CLAHE → Fuzzy C-Means segmentation.

Training Data: INSAT-3D dual-source (IR + RAW) satellite imagery.

Backend: Flask + Gunicorn.

Deployment: Railway.app

Framework: PyTorch

**PROJECT STRUCTURE**:
------------------------------------------
```
├── app.py
├── train.py
├── validate.py
├── dataset.py
├── model/
│   └── fh_odcnn_best.pth
├── data/
│   └── raw/
│       ├── ir_images/
│       ├── raw_images/
│       └── labels.csv
├── static/
│   ├── uploads/
│   └── predictions/
├── templates/
│   └── index.html
└── requirements.txt
```
