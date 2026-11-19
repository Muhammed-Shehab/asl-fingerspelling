# Real-Time ASL Fingerspelling Recognition

Recognizes American Sign Language letters (A-Z, plus space and delete gestures) from a live webcam
and builds them into text. MediaPipe extracts 21 hand landmarks per frame, and a small neural
network (MLP) classifies them. Hold a letter for about 1 second to type it.

Live app: [add the Streamlit link here after deployment]

## Files

- app_web.py - Streamlit web app (live WebRTC camera and photo mode)
- app.py - local-only version using OpenCV (uses the macOS speech command)
- step4_extract_landmarks.py - converts dataset images into landmark features
- step5_train.py - trains and evaluates the classifiers
- models/letter_model.joblib - trained model
- sample_data.csv - sample of the landmark dataset
- results/confusion_matrix.png - test-set confusion matrix

## Run locally (Python 3.11)

    pip install -r requirements.txt
    streamlit run app_web.py

## Retrain (needs the ASL Alphabet dataset from Kaggle in data/asl_raw)

    pip install -r requirements-training.txt
    python step4_extract_landmarks.py
    python step5_train.py

Dataset: ASL Alphabet by "grassknoted" on Kaggle. No API keys or credentials are stored in this repository.
