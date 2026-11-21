# Real-Time ASL Fingerspelling Recognition

A web app that recognizes American Sign Language (ASL) letters from a live webcam and builds them
into text. MediaPipe extracts 21 hand landmarks per frame, and a small neural network (MLP)
classifies them. Hold a letter steady for about 1 second to type it.

**Live app:** https://asl-fingerspelling-8aqvzsxbty2vvscj9vxmhu.streamlit.app/

## Features

- Recognizes 26 letters (A-Z) plus two control gestures (space and delete)
- Real-time recognition from a webcam, running on a laptop CPU (no GPU needed)
- Hold-to-confirm spelling with temporal smoothing, so words can be typed without flicker
- Space, Delete, Clear, and Speak buttons (Speak uses the browser's built-in voice)
- Photo mode: take or upload one photo and get the predicted letter (a fallback when live video cannot connect)
- FPS and per-frame latency overlay on the video

## How it works

1. The camera frame goes to MediaPipe Hands, which returns 21 landmarks (x, y, z), 63 values in total.
2. The landmarks are normalized: the wrist becomes the origin, and the coordinates are divided by the hand size. This removes the effect of hand position and distance from the camera.
3. An MLP classifier (two hidden layers of 128 and 64 units) outputs a probability for each of 28 classes.
4. A frame counts only if its confidence is at least 0.70. A letter becomes "stable" when it wins at least 5 of the last 7 frames.
5. A stable letter must be held for 1.0 second to be typed. To type the same letter twice, lower the hand for 0.4 seconds in between.

## Results

| Item | Value |
|---|---|
| Dataset | ASL Alphabet (Kaggle), 1,200 sampled images per class, 28 classes |
| Images with a detected hand | 26,050 of 33,600 (77.5%) |
| Validation accuracy, Random Forest (baseline) | 95.78% |
| Validation accuracy, MLP (deployed model) | 97.31% |
| Test accuracy, MLP (held-out dataset images) | 97.57% (macro F1 0.972) |
| Live webcam accuracy | 90% |
| FPS on a MacBook Air (CPU) | ~30fps |
| FPS on the hosted app | ~30fps |

The test images come from the same signer and setting as the training images, so the dataset accuracy
is optimistic. The live webcam accuracy is the more honest number.

## Limitations

- The training dataset contains essentially one signer in one setting, so accuracy on new signers is likely lower.
- J and Z involve motion, and a single frame cannot capture it. J is often read as I.
- R and U are often confused, since R is U with crossed fingers.
- M and N have fewer usable training samples, because MediaPipe detected a hand in fewer of their photos.
- Letters are recognized one at a time. Continuous signing and full signs are out of scope.

## Repository structure

    app_web.py                   Streamlit web app (live WebRTC camera and photo mode)
    app.py                       Local-only version using OpenCV (uses the macOS speech command)
    step2_webcam_test.py         Webcam and hand-landmark test
    step4_extract_landmarks.py   Converts dataset images into landmark features (CSV)
    step5_train.py               Trains and evaluates the classifiers
    step6_live.py                Live recognition in an OpenCV window
    models/letter_model.joblib   Trained model (MLP pipeline) and class list
    results/                     Confusion matrix and other results
    sample_data.csv              Sample of the landmark dataset (56 rows)
    requirements.txt             Dependencies for running the app
    requirements-training.txt    Extra dependencies for retraining
    packages.txt                 Linux system package for OpenCV on Streamlit Cloud

## Run locally (Python 3.11)

    python3.11 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app_web.py

Click START in the browser and allow camera access.

## Retrain the model

The dataset is not included in this repository. Download "ASL Alphabet" from Kaggle
(https://www.kaggle.com/datasets/grassknoted/asl-alphabet), unzip it into `data/asl_raw/`, then run:

    pip install -r requirements-training.txt
    python step4_extract_landmarks.py
    python step5_train.py

Step 4 reads about 33,600 images and takes a while, but it only needs to run once.

## Sample data and data dictionary

`sample_data.csv` holds two records per class (56 rows).

| Column | Description |
|---|---|
| label | Sign class: A-Z, space, or del |
| idx | Photo number from the original file name (used to split the data) |
| f0 ... f62 | Normalized landmark coordinates in the order x0, y0, z0, x1, y1, z1, ..., x20, y20, z20 |

The first three feature columns (f0-f2, the wrist) are always 0, because the wrist is the origin
after normalization.

## Deployment (Streamlit Community Cloud)

- Select Python 3.11 under Advanced settings in the deploy dialog.
- `requirements.txt` pins the exact versions used for training. This matters because a scikit-learn
  model may fail to load on a different version.
- `packages.txt` installs the Linux library that OpenCV needs.
- Live video uses WebRTC. If it does not connect on some networks, a TURN server may be needed.
  Add its credentials in the app's Secrets settings (never in the code), in this format:

        [turn]
        urls = ["turn:your-turn-server.example.com:3478"]
        username = "your-username"
        credential = "your-password"

- Photo mode works without WebRTC.

## Privacy

In the hosted app, camera video is streamed to the server only to run recognition. It is not stored.
No API keys or credentials are stored in this repository.

## Acknowledgements

- MediaPipe Hands: F. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," arXiv:2006.10214, 2020.
- Dataset: "ASL Alphabet" on Kaggle, by user "grassknoted".
- Built with scikit-learn, OpenCV, Streamlit, and streamlit-webrtc.

## Author

Muhammed Shehab
