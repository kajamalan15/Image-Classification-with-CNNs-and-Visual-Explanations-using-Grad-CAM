import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import numpy as np
from io import BytesIO

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
class_names = ['cat', 'dog']

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model.load_state_dict(torch.load("C:/Users/Kajamalan/Downloads/best_resnet18_model_epoch_6.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Grad-CAM extractor
def generate_gradcam(input_tensor, pred_class):
    grad_cam = LayerGradCam(model, model.layer4[-1])
    attr = grad_cam.attribute(input_tensor, target=pred_class)
    attr = torch.nn.functional.interpolate(attr, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = attr.squeeze().cpu().detach().numpy()
    return heatmap

# Overlay heatmap on original image
def overlay_gradcam(original_img, heatmap):
    # Normalize heatmap
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)

    # Resize to match image
    heatmap = Image.fromarray(heatmap).resize(original_img.size)
    heatmap = np.asarray(heatmap)

    # Apply colormap
    heatmap_colored = plt.cm.jet(heatmap / 255.0)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    blended = Image.blend(original_img, Image.fromarray(heatmap_colored), alpha=0.5)
    return blended

# Streamlit UI
st.title("🐶 Cat vs Dog Classifier with Grad-CAM")
st.write("Upload an image to classify it as a **Cat** or **Dog**, and visualize what the model is focusing on.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][pred.item()] * 100

    st.markdown(f"### Prediction: **{predicted_class.capitalize()}** ({confidence:.2f}%)")

    # Grad-CAM
    heatmap = generate_gradcam(input_tensor, pred.item())
    gradcam_img = overlay_gradcam(image, heatmap)
    st.markdown("### Grad-CAM Visualization")
    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
