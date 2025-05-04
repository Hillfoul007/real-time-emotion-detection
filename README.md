
# Real-Time Emotion Detection with Emojis 😄😢😠

This project is a real-time emotion detection system built using OpenCV, a pre-trained deep learning model (`.h5`), and deployed via Streamlit. It captures facial expressions from webcam input, classifies emotions using a CNN model, and overlays appropriate emojis on the detected faces in real time.

---

## 📌 Features

- 🎥 Live webcam feed integration.
- 🧠 Emotion prediction using a Keras `.h5` model.
- 😊 Automatic emoji overlay based on detected emotion.
- 💻 Streamlit interface for quick and easy web deployment.
- 🧪 Clean and modular code suitable for experimentation and learning.

---

## 📁 Folder Structure

```
llm-chatbot-python/
│
├── real_time_emotion.py       # Core script to detect emotion and display webcam feed with emojis
├── app.py                     # Streamlit UI entry point
├── face_model.h5              # Pre-trained Keras model for emotion detection
├── emojis/                    # Emoji images corresponding to emotions
├── assets/                    # (Optional) Screenshots or sample output
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Setup Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/real-time-emotion-detection.git
cd real-time-emotion-detection
```

### Step 2: Create a virtual environment and activate it
```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install the dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit app
```bash
streamlit run app.py
```

> ⚠️ Make sure your webcam is connected and accessible.

---

## 😎 Supported Emotions

- **Happy** 😀
- **Sad** 😢
- **Angry** 😠
- **Surprised** 😲
- **Neutral** 😐

---

## 🧠 Technologies Used

- **Python 3.11**
- **Keras / TensorFlow**
- **OpenCV**
- **Streamlit**
- **NumPy**
- **Pillow (PIL)**

---

## 📸 Sample Output

> You can include a sample image or gif here in the `assets/` folder.

```
![Demo](assets/demo.gif)
```

---

## 🙋‍♂️ Author

**Chaman Kataria**  
📍 New Delhi, India  
🔗 [Portfolio](https://chamankataria.netlify.app) • [LinkedIn](https://www.linkedin.com/in/chaman-kataria-b41a2a262/) • [GitHub](https://github.com/hillfoul007)

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
