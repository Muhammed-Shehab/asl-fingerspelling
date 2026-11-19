import time
import base64
import subprocess
from collections import deque, Counter
import cv2
import numpy as np
import joblib
import mediapipe as mp
import streamlit as st

CONF_MIN = 0.70      # minimum confidence to count a frame
WINDOW = 7           # smoothing window (frames)
NEED = 5             # votes needed inside the window
HOLD_SEC = 1.0       # hold a letter this long to confirm it
RELEASE_SEC = 0.4    # drop your hand this long to type the same letter again

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="ASL Spelling", layout="wide")


@st.cache_resource
def load_model():
    b = joblib.load("models/letter_model.joblib")
    return b["model"], list(b["classes"])


@st.cache_resource
def load_hands():
    return mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                          model_complexity=0, min_detection_confidence=0.6,
                          min_tracking_confidence=0.5)


@st.cache_resource
def get_cap():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def normalize(hl):
    pts = np.array([[p.x, p.y, p.z] for p in hl.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        return None
    pts /= scale
    return pts.flatten()


ss = st.session_state
defaults = {"text": "", "hold_label": None, "hold_start": 0.0,
            "locked": None, "none_since": None,
            "history": deque(maxlen=WINDOW)}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v


def apply_label(label):
    if label == "space":
        ss.text += " "
    elif label == "del":
        ss.text = ss.text[:-1]
    else:
        ss.text += label


model, classes = load_model()
hands = load_hands()

st.title("Real-time ASL Fingerspelling")

with st.sidebar:
    st.header("Controls")
    camera_on = st.toggle("Camera on", value=True)
    b_space = st.button("Space")
    b_del = st.button("Delete")
    b_clear = st.button("Clear")
    b_speak = st.button("Speak")
    st.caption("Hold a letter steady for about 1 second to add it. "
               "To type the same letter twice, lower your hand briefly in between.")

if b_space:
    ss.text += " "
if b_del:
    ss.text = ss.text[:-1]
if b_clear:
    ss.text = ""
if b_speak and ss.text.strip():
    subprocess.Popen(["say", ss.text])

left, right = st.columns([3, 2])
with left:
    video_ph = st.empty()
with right:
    letter_ph = st.empty()
    progress_ph = st.empty()
    text_ph = st.empty()


def render_side(shown, progress):
    letter_ph.markdown(f"## Current letter: **{shown or '-'}**")
    progress_ph.progress(float(min(max(progress, 0.0), 1.0)))
    t = ss.text.replace(" ", "\u00a0") + "▌"
    text_ph.markdown(f"### Text\n# {t}")


if not camera_on:
    get_cap().release()
    get_cap.clear()
    video_ph.info("Camera is off. Turn it on in the sidebar.")
    render_side(None, 0)
    st.stop()

cap = get_cap()
if not cap.isOpened():
    st.error("Could not open the camera. Check System Settings > Privacy & Security > Camera.")
    st.stop()

frame_i = 0
prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        st.error("Failed to read from the camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    disp = rgb.copy()

    if res.multi_hand_landmarks:
        hl = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            disp, hl, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style())
        feats = normalize(hl)
        if feats is not None:
            proba = model.predict_proba(feats.reshape(1, -1))[0]
            i = int(np.argmax(proba))
            conf = float(proba[i])
            ss.history.append(classes[i] if conf >= CONF_MIN else None)
        else:
            ss.history.append(None)
    else:
        ss.history.append(None)

    stable = None
    votes = Counter(h for h in ss.history if h is not None)
    if votes:
        label, n = votes.most_common(1)[0]
        if n >= NEED:
            stable = label

    now = time.time()
    added = False
    if stable is None:
        ss.hold_label = None
        if ss.none_since is None:
            ss.none_since = now
        if now - ss.none_since >= RELEASE_SEC:
            ss.locked = None
    else:
        ss.none_since = None
        if stable != ss.locked:
            if stable != ss.hold_label:
                ss.hold_label = stable
                ss.hold_start = now
            elif now - ss.hold_start >= HOLD_SEC:
                apply_label(stable)
                ss.locked = stable
                ss.hold_label = None
                added = True

    progress = 0.0
    if ss.hold_label is not None and stable == ss.hold_label:
        progress = (now - ss.hold_start) / HOLD_SEC

    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
    prev = now
    cv2.putText(disp, f"FPS {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    ok2, buf = cv2.imencode(".jpg", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode()
    video_ph.markdown(f'<img src="data:image/jpeg;base64,{b64}" width="640">', unsafe_allow_html=True)
    frame_i += 1
    if added or frame_i % 3 == 0:
        render_side(stable, progress)
