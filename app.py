import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import tensorflow as tf
import numpy as np

# Suppress TensorFlow logging (optional)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model once and cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_model.h5")

model = load_model()

# Use exact 7 classes from your training data in order
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preprocessing function
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    normalized = face / 255.0
    input_tensor = np.expand_dims(normalized, axis=-1)  # (48,48,1)
    input_tensor = np.expand_dims(input_tensor, axis=0) # (1,48,48,1)
    return input_tensor

# Video processor class
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess and predict
        input_tensor = preprocess(img)
        predictions = model.predict(input_tensor, verbose=0)
        label = class_names[np.argmax(predictions)]

        # Overlay label on frame
        cv2.putText(img, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.header("Real-time Facial Emotion Recognition")

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
