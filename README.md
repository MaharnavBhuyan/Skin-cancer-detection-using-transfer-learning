# Skin-cancer-detection-using-transfer-learning
# 🔬 Skin Cancer Detection using TensorFlow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/skin-cancer-detection/blob/main/Skin_Cancer_Detection_Colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

A deep learning model that classifies skin lesion images as **benign** or **malignant** using **transfer learning (EfficientNetB0)** in TensorFlow/Keras.

## 📌 Overview
- **Task:** Binary image classification (benign vs malignant)
- **Model:** EfficientNetB0 (ImageNet pretrained) + custom classification head
- **Training:** Two phases: (1) train the head with the base frozen, (2) fine-tune the top 30 layers at a low learning rate
- **Techniques:** Data augmentation, class weighting, early stopping, learning-rate scheduling
- **Metrics:** Accuracy, Precision, Recall, AUC, confusion matrix, ROC curve

## 📂 Dataset
[Skin Cancer: Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) (Kaggle). It is downloaded automatically in the notebook via `kagglehub`.

```
data/
├── train/ (benign, malignant)
└── test/  (benign, malignant)
```

## 🖼️ Sample Images and Class Distribution
| Class Distribution | Sample Images |
|---|---|
| ![Class Distribution](images/class_distribution.png) | ![Samples](images/sample_images.png) |

## 🏗️ Model Architecture
```
Input (224×224×3)
  → Data Augmentation (flip, rotation, zoom, contrast)
  → EfficientNetB0 (pretrained, top layers fine-tuned)
  → GlobalAveragePooling2D
  → BatchNorm → Dropout(0.4)
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(1, Sigmoid)
```

## 📈 Results
**Training curves**

![Training Curves](images/training_curves.png)

| Confusion Matrix | ROC Curve |
|---|---|
| ![Confusion Matrix](images/confusion_matrix.png) | ![ROC](images/roc_curve.png) |

**Sample test predictions** (green = correct, red = wrong)

![Predictions](images/test_predictions.png)

| Metric | Test Score |
|---|---|
| Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| AUC | X.XXX |

## 🚀 How to Run

**Google Colab (recommended)**
1. Click the **Open in Colab** badge above.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all.

**Locally**
```bash
git clone https://github.com/<your-username>/skin-cancer-detection.git
cd skin-cancer-detection
pip install tensorflow pandas numpy matplotlib scikit-learn kagglehub
python skin_cancer_detection.py
```

## 🛠️ Tech Stack
Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · scikit-learn

## 📁 Repository Structure
```
├── Skin_Cancer_Detection_Colab.ipynb   # Colab notebook
├── skin_cancer_detection.py            # Script version
├── images/                             # Result plots used in this README
├── .gitignore
└── README.md
```

## ⚠️ Disclaimer
This project is for **educational purposes only** and is not a medical diagnostic tool. Always consult a qualified dermatologist.

## 👤 Author
**Maharnav Bhuyan**
B.Tech CSE, SRM Institute of Science and Technology, Chennai

