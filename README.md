# COVID-19 Chest X-ray Multi-Class Classification with Grad-CAM

CNN-based multi-class classification of chest X-ray images (Normal, COVID, Viral Pneumonia) with Grad-CAM visualization for model interpretation.

## 🎯 Overview

Classifies chest X-rays into three categories:
- **Normal**: Healthy chest X-rays
- **COVID**: COVID-19 infections  
- **Viral Pneumonia**: Viral pneumonia infections

## 📊 Dataset

**Download**: [COVID-19 Image Dataset - Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/data)

Expected structure:
```
DATASETS/Covid19-dataset/
├── train/
│   ├── Covid/
│   ├── Normal/
│   └── Viral Pneumonia/
└── test/
    ├── Covid/
    ├── Normal/
    └── Viral Pneumonia/
```

## 🚀 Features

- Multi-class CNN classification
- Grad-CAM visualization for model interpretability
- Confusion matrix and performance metrics
- Visual comparison of correct/incorrect predictions

## �️ Dependencies

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn seaborn
```

## 🔧 Usage

1. Download dataset from Kaggle link above
2. Update dataset paths in notebook
3. Run `chest-xray-covid-classification-gradcam.ipynb` sequentially
4. View Grad-CAM visualizations and performance metrics

## 🏗️ Model Architecture

- Input: 224×224×3 images
- 3 Conv2D layers (64, 32, 16 filters) + MaxPooling
- Dense layer (64 neurons) + Softmax output (3 classes)
- Loss: Sparse categorical crossentropy

## 📈 Results

### Model Performance
- **Overall Accuracy**: ~85-90% on test set
- **COVID**: Precision: 0.92, Recall: 0.85, F1-score: 0.88
- **Normal**: Precision: 0.89, Recall: 0.94, F1-score: 0.91
- **Viral Pneumonia**: Precision: 0.81, Recall: 0.78, F1-score: 0.79

### ⚠️ Important Findings - Model Limitation
**Critical Issue Identified**: The Grad-CAM analysis reveals that the model is **NOT focusing on clinically relevant lung regions**. Instead, it concentrates on:
- Background areas outside the lung region
- Image borders and artifacts
- Non-anatomical features

This indicates the model may be learning from dataset biases rather than meaningful medical features, raising concerns about its clinical reliability and generalizability.

### Visualizations
- Confusion matrix showing class-wise performance
- Grad-CAM heatmaps revealing attention patterns
- Comparison of correct vs incorrect predictions

---

**Note**: For educational/research purposes only. Clinical use requires additional validation.
