# Potato Leaf Disease Detection App

This application detects potato leaf diseases (Early Blight, Late Blight, or Healthy) using a trained deep learning model. It allows users to upload images, capture live images via webcam, or analyze real-time video streams for disease prediction. The app also provides visual Grad-CAM heatmaps to highlight areas of focus for predictions.

---

## Features

1. **Upload Image**
   - Upload a potato leaf image for classification.
   - Displays predictions from three models: Custom CNN, Inception, and ResNet.
   - Shows Grad-CAM heatmap overlay for better interpretability.

2. **Use Camera**
   - Capture images directly using a webcam for classification.
   - Predicts disease type and displays model confidence.

3. **Real-Time Video Analysis**
   - Analyze live webcam video streams to detect diseases.
   - Shows bounding boxes dynamically focusing on activated regions.
   - Confirms prediction after analyzing 15 frames for stability.

4. **About Section**
   - Detailed description of each feature and models used in the app.

---

## Models Used
1. **Custom CNN Model**: Lightweight model designed for quick predictions.
2. **Inception Model**: Pre-trained model fine-tuned for potato leaf disease classification.
3. **ResNet Model**: Another pre-trained model for robust classification.

---

