import streamlit as st
def about():
    st.title("About This Project")

    st.write("""
        This project is designed to perform **real-time potato leaf disease detection** 
        using advanced machine learning and computer vision techniques. It incorporates 
        various features and functionalities to enhance the user experience and improve 
        the model's reliability. Below is a detailed explanation of each feature used in this project:
    """)

    st.subheader("📸 Real-Time Video Stream")
    st.write("""
        - The system captures a live video feed from the webcam and processes each frame in real-time.
        - A **bounding box** dynamically highlights the region of the image being analyzed by the model.
        - The model runs predictions on each frame to classify the potato leaf as healthy or diseased.
        - Users can control the video feed using **Play**, **Pause**, and **Stop** buttons.
    """)

    st.subheader("🎥 Camera Input")
    st.write("""
        - Users can capture an image of a potato leaf using their webcam.
        - The captured image is preprocessed and passed through three different models for predictions:
            - **Custom CNN Model**: A specialized model trained on the dataset.
            - **Inception Model**: A pre-trained InceptionV3 model fine-tuned for the task.
            - **ResNet Model**: A pre-trained ResNet50 model fine-tuned for the task.
        - The predictions from all three models, along with their confidence scores, are displayed for comparison.
    """)

    st.subheader("🔥 Grad-CAM Visualization")
    st.write("""
        - Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which regions of the leaf 
          the model focuses on for its predictions.
        - While real-time Grad-CAM overlays can slow down video processing, this feature is critical 
          for understanding model behavior.
        - The **bounding box** is dynamically calculated based on the heatmap regions, helping users 
          identify the model's area of focus.
    """)

    st.subheader("🔲 Bounding Box for Classification Area")
    st.write("""
        - A **bounding box** dynamically highlights the region of the potato leaf where the 
          model focuses its analysis.
        - This bounding box is calculated using the Grad-CAM heatmap and dynamically adjusts 
          to the most activated regions in the image.
    """)

    st.subheader("🛠️ Advanced Controls")
    st.write("""
        - The video feed includes **Play**, and **Stop** buttons for user control.
        - Frames are analyzed in real-time, but after every **15 frames**, the system confirms 
          predictions with a bounding box and the most probable answer for improved reliability.
    """)

    st.subheader("🌟 Multi-Model Integration")
    st.write("""
        - The project uses multiple models (Custom CNN, Inception, ResNet) to provide robust predictions.
        - This ensures better accuracy and allows for model comparison.
    """)

    st.subheader("📊 Dashboard-Like Functionality")
    st.write("""
        - The application is designed like a dashboard, providing real-time insights and visualizations.
        - Users can navigate between features such as **Real-Time Detection**, **Capture Image**, 
          and **About Page** with ease.
    """)

    st.subheader("🚀 Technologies Used")
    st.write("""
        - **TensorFlow**: For training and deploying the machine learning models.
        - **OpenCV**: For image and video processing.
        - **Streamlit**: For creating an interactive and user-friendly web application.
        - **Grad-CAM**: For heatmap-based visualizations of model activations.
        - **Custom Trained CNN Models**:Shows Use of CNN Architecture    
        - **Pre-Trained Models**: Transfer learning from state-of-the-art architectures (Inception and ResNet).
    """)

    st.info("This application demonstrates the power of AI in agriculture by aiding in the early detection of potato leaf diseases, helping farmers make informed decisions.")

