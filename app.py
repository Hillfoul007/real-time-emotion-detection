import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import tensorflow as tf
import numpy as np

st.title("Real-Time Emotion Detection")
# Load model


@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?export=download&id=1cSKFLMk-qoUDRzDszWf0iPdIAhMrwJdF"
    model_path = "face_model.h5"

    if not os.path.exists(model_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("Failed to download the model from Google Drive.")
            return None

    return tf.keras.models.load_model(model_path)

model = load_model()


class_names = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Disgusted']

# Preprocessing function
def preprocess(frame):
    face = cv2.resize(frame, (48, 48))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

# Video processor
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        input_tensor = preprocess(img)
        predictions = model.predict(input_tensor)
        label = class_names[np.argmax(predictions)]
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream from webcam
st.header("Real-time Emotion Detection")
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
