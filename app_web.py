import json
import threading
import time
from collections import deque, Counter

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---- settings you can tune ----
CONF_MIN = 0.70      # minimum confidence for a frame to count
WINDOW = 7           # smoothing window (frames)
NEED = 5             # votes needed inside the window
HOLD_SEC = 1.0       # hold a letter this long to type it
RELEASE_SEC = 0.4    # drop your hand this long to type the same letter again
MODEL_PATH = "models/letter_model.joblib"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="ASL Fingerspelling", layout="wide")


def normalize(hl):
    # must match the normalization used in training
    pts = np.array([[p.x, p.y, p.z] for p in hl.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        return None
    pts /= scale
    return pts.flatten()


@st.cache_resource
def load_model():
    b = joblib.load(MODEL_PATH)
    return b["model"], list(b["classes"])


@st.cache_resource
def get_static_hands():
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                          min_detection_confidence=0.4)


class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        b = joblib.load(MODEL_PATH)
        self.model = b["model"]
        self.classes = list(b["classes"])
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.lock = threading.Lock()
        self.text = ""
        self.history = deque(maxlen=WINDOW)
        self.hold_label = None
        self.hold_start = 0.0
        self.locked = None
        self.none_since = None
        self.shown = None
        self.progress = 0.0
        self.fps = 0.0
        self.prev_t = time.time()

    def _apply(self, label):
        if label == "space":
            self.text += " "
        elif label == "del":
            self.text = self.text[:-1]
        else:
            self.text += label

    def command(self, cmd):
        with self.lock:
            if cmd == "space":
                self.text += " "
            elif cmd == "del":
                self.text = self.text[:-1]
            elif cmd == "clear":
                self.text = ""

    def get_state(self):
        with self.lock:
            return self.text, self.shown, self.progress

    def recv(self, frame):
        t0 = time.time()
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        pred = None
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                img, hl, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())
            feats = normalize(hl)
            if feats is not None:
                proba = self.model.predict_proba(feats.reshape(1, -1))[0]
                i = int(np.argmax(proba))
                if float(proba[i]) >= CONF_MIN:
                    pred = self.classes[i]

        now = time.time()
        dt = now - self.prev_t
        self.prev_t = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        with self.lock:
            self.history.append(pred)
            stable = None
            votes = Counter(h for h in self.history if h is not None)
            if votes:
                label, n = votes.most_common(1)[0]
                if n >= NEED:
                    stable = label

            progress = 0.0
            if stable is None:
                self.hold_label = None
                if self.none_since is None:
                    self.none_since = now
                if now - self.none_since >= RELEASE_SEC:
                    self.locked = None
            else:
                self.none_since = None
                if stable != self.locked:
                    if stable != self.hold_label:
                        self.hold_label = stable
                        self.hold_start = now
                    progress = (now - self.hold_start) / HOLD_SEC
                    if progress >= 1.0:
                        self._apply(stable)
                        self.locked = stable
                        self.hold_label = None
                        progress = 0.0

            self.shown = stable
            self.progress = progress
            text = self.text

        # overlay drawn on the video itself
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(img, stable or "-", (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0) if stable else (130, 130, 130), 4)
        cv2.rectangle(img, (130, 25), (330, 45), (255, 255, 255), 1)
        cv2.rectangle(img, (130, 25), (130 + int(200 * min(progress, 1.0)), 45),
                      (0, 255, 0), -1)
        cv2.putText(img, text[-26:] + "_", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        ms = (time.time() - t0) * 1000
        cv2.putText(img, f"{self.fps:.0f} FPS  {ms:.0f} ms", (w - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def rtc_config():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:  # optional TURN server, read from Streamlit secrets (never from the code)
        t = st.secrets["turn"]
        urls = t["urls"]
        if isinstance(urls, str):
            urls = [urls]
        servers.append({"urls": list(urls), "username": t["username"],
                        "credential": t["credential"]})
    except Exception:
        pass
    return {"iceServers": servers}


model, classes = load_model()
ss = st.session_state
if "text" not in ss:
    ss.text = ""

st.title("Real-time ASL Fingerspelling")
tab_live, tab_photo = st.tabs(["Live spelling", "Photo mode"])

with tab_live:
    left, right = st.columns([3, 2])
    with left:
        ctx = webrtc_streamer(
            key="asl",
            video_processor_factory=ASLProcessor,
            rtc_configuration=rtc_config(),
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False},
            async_processing=True,
        )
        st.caption("Click START and allow the camera. Hold a letter steady for "
                   "about 1 second to type it. To type the same letter twice, "
                   "lower your hand briefly in between.")
    with right:
        letter_ph = st.empty()
        prog_ph = st.empty()
        text_ph = st.empty()
        c1, c2, c3 = st.columns(3)
        b_space = c1.button("Space")
        b_del = c2.button("Delete")
        b_clear = c3.button("Clear")
        speak_ph = st.empty()

with tab_photo:
    st.write("Show a letter to the camera, take a photo, and the model predicts it. "
             "This mode works even when live video cannot connect.")
    shot = st.camera_input("Take a photo of a letter")
    if shot is not None:
        data = np.frombuffer(shot.getvalue(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        res = get_static_hands().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            st.warning("No hand detected. Try again with your whole hand in view.")
        else:
            hl = res.multi_hand_landmarks[0]
            feats = normalize(hl)
            proba = model.predict_proba(feats.reshape(1, -1))[0]
            order = np.argsort(proba)[::-1][:3]
            st.success(f"Predicted: {classes[order[0]]}  ({proba[order[0]] * 100:.0f}% confident)")
            st.caption("Top 3: " + ", ".join(f"{classes[i]} {proba[i] * 100:.0f}%" for i in order))


def render_side(shown, progress, text):
    letter_ph.markdown(f"### Current letter: **{shown or '-'}**")
    prog_ph.progress(float(min(max(progress, 0.0), 1.0)))
    text_ph.markdown("### Text\n## " + text.replace(" ", "\u00a0") + "▌")


def render_speak(text):
    safe = json.dumps(text)
    html = ("<button style='padding:8px 18px;font-size:16px;cursor:pointer' "
            "onclick='var u=new SpeechSynthesisUtterance(" + safe + ");"
            "window.speechSynthesis.cancel();window.speechSynthesis.speak(u);'>"
            "Speak</button>")
    with speak_ph.container():
        components.html(html, height=55)


proc = ctx.video_processor if ctx.state.playing else None
if b_space or b_del or b_clear:
    cmd = "space" if b_space else ("del" if b_del else "clear")
    if proc is not None:
        proc.command(cmd)
    elif cmd == "space":
        ss.text += " "
    elif cmd == "del":
        ss.text = ss.text[:-1]
    else:
        ss.text = ""

if proc is not None:
    ss.text = proc.get_state()[0]
render_side(None, 0.0, ss.text)
render_speak(ss.text)

last_text = ss.text
while ctx.state.playing:
    p = ctx.video_processor
    if p is not None:
        text, shown, progress = p.get_state()
        ss.text = text
        render_side(shown, progress, text)
        if text != last_text:
            render_speak(text)
            last_text = text
    time.sleep(0.15)
