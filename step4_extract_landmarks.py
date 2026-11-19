import os
import re
import sys
import csv
import random
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

TRAIN_DIR = "data/asl_raw/asl_alphabet_train/asl_alphabet_train"
OUT_CSV = "data/landmarks.csv"
PER_LETTER = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
SEED = 42
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "del"]


def normalize(hand_landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]  # wrist becomes the origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        return None
    pts /= scale
    return pts.flatten()  # 63 numbers


def main():
    random.seed(SEED)
    os.makedirs("data", exist_ok=True)
    mp_hands = mp.solutions.hands
    stats = {}
    header = ["label", "idx"] + [f"f{i}" for i in range(63)]

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as hands, \
         open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for letter in LETTERS:
            folder = os.path.join(TRAIN_DIR, letter)
            files = sorted(x for x in os.listdir(folder) if x.lower().endswith((".jpg", ".jpeg", ".png")))
            chosen = random.sample(files, min(PER_LETTER, len(files)))
            ok_count = 0
            for name in tqdm(chosen, desc=f"Class {letter}", leave=False):
                img = cv2.imread(os.path.join(folder, name))
                if img is None:
                    continue
                img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_REPLICATE)
                img = cv2.resize(img, None, fx=2, fy=2)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if not res.multi_hand_landmarks:
                    continue
                feats = normalize(res.multi_hand_landmarks[0])
                if feats is None:
                    continue
                m = re.search(r"(\d+)", name)
                idx = int(m.group(1)) if m else -1
                writer.writerow([letter, idx] + [f"{v:.5f}" for v in feats])
                ok_count += 1
            stats[letter] = (ok_count, len(chosen))
            f.flush()
            print(f"{letter}: {ok_count}/{len(chosen)} hands detected")

    total_ok = sum(v[0] for v in stats.values())
    total = sum(v[1] for v in stats.values())
    print(f"\nTOTAL: {total_ok}/{total} detected ({100 * total_ok / total:.1f}%)")
    print(f"Saved to {OUT_CSV}")


main()
