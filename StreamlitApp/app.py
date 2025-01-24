import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
from Home import home
from About import about
import time 
from collections import Counter
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "custom_model.h5")
MODEL_PATH2 = os.path.join(BASE_DIR, "model", "Inception.h5")
MODEL_PATH3 = os.path.join(BASE_DIR, "model", "ResNet.h5")

# Load the trained model
# MODEL_PATH = 'model/custom_model.h5'
# MODEL_PATH2 = 'model/Inception.h5'
# MODEL_PATH3 = 'model/ResNet.h5'

model = tf.keras.models.load_model(MODEL_PATH)
model2 = tf.keras.models.load_model(MODEL_PATH2)
model3 = tf.keras.models.load_model(MODEL_PATH3)

class_names =["Early Blight","Late Blight","Healthy"]

def realTime():
    st.write("Ensure your webcam is connected.")

    if "video_state" not in st.session_state:
        st.session_state.video_state = "stopped"  
    if "frame_predictions" not in st.session_state:
        st.session_state.frame_predictions = [] 

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Play"):
            st.session_state.video_state = "playing"


    with col2:
        if st.button("Stop"):
            st.session_state.video_state = "stopped"

    FRAME_WINDOW = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access the camera. Please check if the webcam is connected or being used by another application.")
        return
    
    frame_rate = 10
    frame_delay = 1 / frame_rate
    frame_count = 0

    while st.session_state.video_state != "stopped":
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from the webcam. Please refresh the app or check the camera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame for the model
        img_array = preprocess_frame(frame)
        predictions = model2.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]

        st.session_state.frame_predictions.append((class_idx, confidence))
        frame_count += 1


        # Generate Grad-CAM heatmap
        heatmap = grad_cam(model2, img_array, "conv2d_116")  
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        threshold = 0.67  # Adjust as needed
        activated_region = heatmap_resized > threshold

        # Find the coordinates of the bounding box
        y_indices, x_indices = np.where(activated_region)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            x_min = max(0, x_min)
            x_max = min(frame.shape[1], x_max)
            y_min = max(0, y_min)
            y_max = min(frame.shape[0], y_max)

            # Draw bounding box on the frame
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (200, 0, 0), 2  # Blue color with thickness 2
            )
        else:
            st.warning("No activated regions detected.")

        # Display video frame with bounding box
        FRAME_WINDOW.image(frame, channels="RGB")

        if frame_count == 15:
            predictions_count = Counter([pred[0] for pred in st.session_state.frame_predictions])
            most_common_class_idx, count = predictions_count.most_common(1)[0]
            avg_confidence = np.mean([pred[1] for pred in st.session_state.frame_predictions if pred[0] == most_common_class_idx])

            st.success(f"Confirmed Prediction: {class_names[most_common_class_idx]} ({avg_confidence * 100:.2f}%)")
            st.session_state.frame_predictions = []
            frame_count = 0

        time.sleep(frame_delay)


    # Release the camera resource
    cap.release()
    FRAME_WINDOW.empty()
    st.write("Video stream stopped.")
    
def upload():
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        predictions2 = model2.predict(img_array)
        class_idx2 = np.argmax(predictions2[0])

        predictions3 = model3.predict(img_array)
        class_idx3 = np.argmax(predictions3[0])

        st.subheader("Prediction Results")
        
        st.write("### Custom CNN Model")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

        st.write("### Inception Model")
        st.success(f"Prediction: {class_names[class_idx2]}")
        st.info(f"Confidence: {predictions2[0][class_idx2]*100:.2f}")

        st.write("### ResNet Model")
        st.success(f"Prediction: {class_names[class_idx3]}")
        st.info(f"Confidence: {predictions3[0][class_idx3]*100:.2f}")

        st.subheader("Generating Grad Cam")

        # Generate Grad-CAM heatmap
        heatmap = grad_cam(model2, img_array,"conv2d_116")
        result_img = overlay_heatmap_Img(heatmap, img, opacity=0.4)
        st.image(result_img,caption="Grad-CAM Result", use_container_width=True)

def camera():
    camera_image = st.camera_input("Capture a potato leaf image")
    if camera_image is not None:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image",use_container_width=True)
        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        predictions2 = model2.predict(img_array)
        class_idx2 = np.argmax(predictions2[0])

        predictions3 = model3.predict(img_array)
        class_idx3 = np.argmax(predictions3[0])


        st.subheader("Prediction Results")
        
        st.write("### Custom CNN Model")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

        st.write("### Inception Model")
        st.success(f"Prediction: {class_names[class_idx2]}")
        st.info(f"Confidence: {predictions2[0][class_idx2]*100:.2f}")

        st.write("### ResNet Model")
        st.success(f"Prediction: {class_names[class_idx3]}")
        st.info(f"Confidence: {predictions3[0][class_idx3]*100:.2f}")
        

# Function to preprocess uploaded images
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def preprocess_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])  # Predicted class index
        loss = predictions[:, predicted_class]

    gradients = tape.gradient(loss, conv_outputs)
    
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    
    # Apply the pooled gradients to the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:-1])

    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_gradients[i] * conv_outputs[:, :, i]

    # Apply ReLU to focus on positive influence only
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap for visualization
    heatmap /= np.max(heatmap)
    return heatmap


# Overlay heatmap on the frame
def overlay_heatmap(heatmap, frame,opacity):


    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_frame = cv2.addWeighted(frame, 1-opacity, heatmap, opacity, 0)
    return superimposed_frame

def overlay_heatmap_Img(heatmap,img,opacity):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (OpenCV uses BGR)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - opacity, heatmap, opacity, 0)

    return superimposed_img



if __name__=="__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home()
    
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose Your Work", ["Upload Image", "Real Time Video","About"],index=None)
    
    if(option=="Upload Image"):
        upload()
        
    # elif(option=="Use Camera"):
    #     camera()
        
    elif(option=="Real Time Video"):
        realTime()
    elif(option=="About"):
        about()
    
    st.markdown("---")
    st.info("📌 Navigate to different sections using the sidebar.")
    st.write("Made with ❤️ by Shubham Srivastava")
