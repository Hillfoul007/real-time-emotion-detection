# Real-Time Emotion Detection

This project detects human emotions in real-time using webcam feed and displays appropriate emojis.

## Features

- Face detection using OpenCV
- Emotion classification using a trained CNN model (`face_model.h5`)
- Real-time emoji overlay
- Web interface powered by Flask

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/Hillfoul007/real-time-emotion-detection.git
    cd real-time-emotion-detection
    ```

2. Create and activate virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python app.py
    ```

## Requirements

See `requirements.txt`.

## Note

Do **not** push the `venv` or large model files (`.pyd`, `.dll`, etc.) to GitHub. Use `.gitignore` to exclude them.
