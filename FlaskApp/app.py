from flask import Flask, render_template, Response,request, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from collections import Counter
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "custom_model.h5")
MODEL_PATH2 = os.path.join(BASE_DIR, "model", "Inception.h5")
MODEL_PATH3 = os.path.join(BASE_DIR, "model", "ResNet.h5")

custom_model = tf.keras.models.load_model(MODEL_PATH)
inception_model = tf.keras.models.load_model(MODEL_PATH2)
resnet_model = tf.keras.models.load_model(MODEL_PATH3)

class_names = ["Early Blight", "Late Blight", "Healthy"]

frame_predictions = []
frame_count = 0
frame_rate = 10
frame_delay = 1 / frame_rate

# Preprocess image
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def preprocess_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Generate Grad-CAM heatmap
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    gradients = tape.gradient(loss, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:-1])

    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_gradients[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Overlay heatmap on image
def overlay_heatmap_img(heatmap, img, opacity=0.4):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - opacity, heatmap, opacity, 0)
    return superimposed_img


def generate_frames():
    global is_paused
    while True:
        if not is_paused:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            input_frame = preprocess_frame(rgb_frame)

            predictions = inception_model.predict(input_frame)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            prediction_text = f"Prediction: {class_names[class_idx]} ({confidence * 100:.2f}%)"

            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            
            continue


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and prediction."""
    if 'image' not in request.files:
        return render_template('error.html', error="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('error.html', error="No file selected.")

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Open image and preprocess
    img = Image.open(file_path)
    img_array = preprocess_image(img)

    # Predictions from models
    predictions = custom_model.predict(img_array)
    class_idx = np.argmax(predictions[0])

    predictions2 = inception_model.predict(img_array)
    class_idx2 = np.argmax(predictions2[0])

    predictions3 = resnet_model.predict(img_array)
    class_idx3 = np.argmax(predictions3[0])

    # Generate Grad-CAM heatmap
    heatmap = grad_cam(inception_model, img_array, "conv2d_116")
    result_img = overlay_heatmap_img(heatmap, img)
    result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grad_cam_result.jpg')
    Image.fromarray(result_img).save(result_img_path)

    # Render results on a new page
    return render_template(
        'results.html',
        custom_model_prediction=class_names[class_idx],
        custom_model_confidence=f"{predictions[0][class_idx] * 100:.2f}%",
        inception_model_prediction=class_names[class_idx2],
        inception_model_confidence=f"{predictions2[0][class_idx2] * 100:.2f}%",
        resnet_model_prediction=class_names[class_idx3],
        resnet_model_confidence=f"{predictions3[0][class_idx3] * 100:.2f}%",
        grad_cam_path=f"/uploads/grad_cam_result.jpg" 
    )


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

is_paused = False  
cap = cv2.VideoCapture(0)

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused  # Toggle pause
    return ("Paused" if is_paused else "Playing", 200)

@app.route("/realtime")
def realtime():
     return render_template('realtime.html')


if __name__ == '__main__':
    app.run(debug=True)
