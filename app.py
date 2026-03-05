import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import InputLayer

# ── Patch for batch_shape deserialization error ──────────────────────────────
class FixedInputLayer(InputLayer):
    """Wraps InputLayer to handle the legacy `batch_shape` config key."""
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs:
            batch_shape = kwargs.pop("batch_shape")
            kwargs["input_shape"] = batch_shape[1:]
            kwargs["batch_size"] = batch_shape[0]
        super().__init__(*args, **kwargs)

# ── RTC configuration (STUN server required for WebRTC on cloud) ─────────────
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "face_model.h5",
        custom_objects={"InputLayer": FixedInputLayer},
    )

model = load_model()
class_names = ['Happy', 'Sad', 'Fear', 'Angry', 'Surprised', 'Neutral', 'Disgusted']

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(frame):
    face = cv2.resize(frame, (48, 48))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

# ── Video processor ───────────────────────────────────────────────────────────
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        input_tensor = preprocess(img)
        predictions = model.predict(input_tensor)
        label = class_names[np.argmax(predictions)]
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎭 Real-time Emotion Detection")
st.write("Allow camera access and wait for the model to load.")

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)