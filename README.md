# 🧠 Image Classification with CNNs & Grad-CAM

This project involves building a deep learning model for image classification (Cats vs Dogs) using Convolutional Neural Networks (CNNs), and applying Explainable AI (XAI) techniques like **Grad-CAM** to visualize how the model makes decisions.

## 🚀 Project Objectives
- Train or fine-tune a CNN (ResNet18) for image classification.
- Evaluate model performance using accuracy and confusion matrix.
- Apply **Grad-CAM** to interpret and visualize model predictions.
- Reflect on the explainability and usefulness of XAI in real-world scenarios.

## 🛠️ Tools & Technologies
- **Python 3.x**
- **PyTorch**, torchvision
- **matplotlib**, **seaborn**
- **pytorch-gradcam** (for Grad-CAM visualizations)
- **Jupyter Notebooks**, **Streamlit** (for deployment)
- Git & GitHub for version control

## 📁 Dataset
- Kaggle Cats vs Dogs (subset) or similar binary dataset.
- ~200–500 images with a 80/20 train-test split.
- Applied image augmentations like resizing, flipping, and normalization.

## 📊 Tasks Overview
1. **Data Preparation**: Augmentation, normalization, visualization.
2. **Model Training**: Fine-tuned ResNet18, trained for several epochs.
3. **Evaluation**: Accuracy, loss curves, confusion matrix, misclassified samples.
4. **Explainability**: Applied Grad-CAM to visualize model focus areas.
5. **Reflections**: Analyzed model decisions and XAI applicability.

### 📌 Sample Output:  
> **Prediction:** Dog 🐶  
> **Grad-CAM:** Highlights facial and ear region

## ✨ Future Improvements
- Expand dataset size
- Experiment with different CNN architectures
- Improve Grad-CAM resolution for finer visualizations
