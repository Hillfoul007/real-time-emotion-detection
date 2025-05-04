import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
# Load the trained model
model = load_model('C:\\Users\\Kataria\\Downloads\\archive\\face_model.h5')

emotion_emoji_dict = {
    0: "😠",  # Angry
    1: "🤢",  # Disgust
    2: "😨",  # Fear
    3: "😄",  # Happy
    4: "😢",  # Sad
    5: "😲",  # Surprise
    6: "😐",  # Neutral
}


def predict_emotion(frame):
    # Convert the image to grayscale (if it's not already)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize image to 48x48 (the size expected by the model)
    frame = cv2.resize(frame, (48, 48))

    # Normalize the image and expand dimensions
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(frame)
    emotion_index = np.argmax(prediction)  # Get index of highest prediction
    return emotion_index


import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('C:\\Users\\Kataria\\Downloads\\archive\\face_model.h5')

# Map predictions to emojis
emotion_emoji_dict = {
    0: "😠",  # Angry
    1: "🤢",  # Disgust
    2: "😨",  # Fear
    3: "😄",  # Happy
    4: "😢",  # Sad
    5: "😲",  # Surprise
    6: "😐",  # Neutral
}


# Create a function to predict emotion
def predict_emotion(frame):
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize image to 48x48
    frame = cv2.resize(frame, (48, 48))
    # Normalize and expand dimensions for prediction
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(frame)
    emotion_index = np.argmax(prediction)  # Get index of highest prediction
    return emotion_index


# Streamlit UI
st.title("Real-Time Mood Detector")

# Upload an image file
image_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Read and display the image
    image = np.array(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Make a prediction
    emotion_index = predict_emotion(image)
    emoji = emotion_emoji_dict[emotion_index]  # Map to emoji

    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Display the predicted emotion with emoji
    st.write(f"Predicted Emotion: {emoji}")
