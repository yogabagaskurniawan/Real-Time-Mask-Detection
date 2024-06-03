import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# Load the model using TensorFlow SavedModel format
model_path = "converted_savedmodel/model.savedmodel"  # Path to the SavedModel directory
loaded_model = tf.saved_model.load(model_path)

# Load labels
with open("converted_savedmodel/labels.txt", "r") as file:
    labels = file.read().split("\n")

# Streamlit application
st.title("Real-Time Mask Detection")
st.text("Click 'Start Camera' to use your webcam")

# Start and stop buttons
start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

# Webcam feed and prediction display
if "camera_started" not in st.session_state:
    st.session_state["camera_started"] = False

if start_button:
    st.session_state["camera_started"] = True

if stop_button:
    st.session_state["camera_started"] = False

if st.session_state["camera_started"]:
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    # List to store the latest 3 predictions
    latest_predictions = []

    # Placeholder for predictions
    predictions_placeholder = st.empty()

    while st.session_state["camera_started"]:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to match model input size
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Display the frame in the Streamlit app
        FRAME_WINDOW.image(frame_rgb)

        # Preprocess the frame
        image = frame_resized.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict the class
        prediction = loaded_model(tf.constant(image), training=False)
        prediction = prediction.numpy()

        # Get top prediction
        top_index = np.argmax(prediction[0])
        top_label = labels[top_index]
        top_confidence = prediction[0][top_index] * 100

        # Update latest predictions list
        latest_predictions.append((top_label, top_confidence))
        if len(latest_predictions) > 3:
            latest_predictions.pop(0)

        # Display latest 3 predictions
        with predictions_placeholder.container():
            st.write("Latest Predictions:")
            for i, (label, confidence) in enumerate(latest_predictions):
                st.write(f"{label} with confidence {confidence:.2f}%")

    cap.release()
