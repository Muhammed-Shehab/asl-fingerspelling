import time
from collections import deque, Counter
import cv2
import numpy as np
import joblib
import mediapipe as mp

CONF_MIN = 0.70   # below this confidence, show nothing
WINDOW = 7        # look at the last 7 frames
NEED = 5          # a letter must win 5 of those 7 to be accepted

bundle = joblib.load("models/letter_model.joblib")
model = bundle["model"]
classes = list(bundle["classes"])

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def normalize(hand_landmarks):
    # identical to the normalization used in training
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        return None
    pts /= scale
    return pts.flatten()


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open the camera.")
        return

    history = deque(maxlen=WINDOW)
    prev = time.time()
    fps = 0.0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        model_complexity=0, min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            top3 = []
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())
                feats = normalize(hl)
                if feats is not None:
                    proba = model.predict_proba(feats.reshape(1, -1))[0]
                    order = np.argsort(proba)[::-1][:3]
                    top3 = [(classes[i], float(proba[i])) for i in order]
                    best, conf = top3[0]
                    history.append(best if conf >= CONF_MIN else None)
                else:
                    history.append(None)
            else:
                history.append(None)

            stable = None
            votes = Counter(h for h in history if h is not None)
            if votes:
                label, n = votes.most_common(1)[0]
                if n >= NEED:
                    stable = label

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)
            shown = stable if stable else "-"
            cv2.putText(frame, shown, (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (0, 255, 0) if stable else (120, 120, 120), 6)
            if top3:
                y = 30
                for lab, p in top3:
                    cv2.putText(frame, f"{lab}: {p * 100:.0f}%", (330, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y += 30
            cv2.putText(frame, f"FPS {fps:.0f}", (500, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Q = quit", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Step 6 - Live ASL letters", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


main()
