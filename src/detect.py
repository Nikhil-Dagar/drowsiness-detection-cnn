"""
detect.py
Real-time drowsiness detection.

Upgrades over a plain EAR-threshold approach:
  1. A trained CNN classifies each cropped eye as Open/Closed
     (EAR is still computed and shown, used only as a lightweight
     cross-check / fallback signal, not the sole decision maker).
  2. Mouth Aspect Ratio (MAR) adds a second, independent signal —
     yawning — instead of only tracking eye closure.
  3. Every drowsy / yawn event is timestamped and logged to CSV.
  4. A non-blocking alarm sound plays on the audio thread so the
     video loop never stutters.

Usage:
    python src/detect.py
Press 'q' to quit.
"""

import csv
import os
import threading
import time
from datetime import datetime

import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

from landmarks import (
    get_face_detector,
    get_landmark_predictor,
    shape_to_np,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    extract_eye_region,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_IDX,
)

# ---- Paths ----
LANDMARK_MODEL_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
CNN_MODEL_PATH = os.path.join("models", "eye_state_cnn.keras")
LOG_PATH = os.path.join("logs", "event_log.csv")
ALARM_SOUND_PATH = os.path.join("assets", "alarm.wav")

# ---- Thresholds (tune these for your webcam / lighting) ----
ALARM_REPEAT_SECONDS = 3   # re-play alarm every 3 sec while still drowsy
EYE_CLOSED_PROB_THRESHOLD = 0.5      # CNN output < this => eyes considered closed
CONSEC_CLOSED_FRAMES = 20            # ~ frames before triggering a drowsy alert
MAR_YAWN_THRESHOLD = 0.6
CONSEC_YAWN_FRAMES = 15


def play_alarm():
    """Plays the alarm sound on a separate thread so video doesn't freeze."""
    try:
        from playsound import playsound
        if os.path.exists(ALARM_SOUND_PATH):
            playsound(ALARM_SOUND_PATH)
    except Exception as e:
        print(f"[warn] could not play alarm sound: {e}")


def log_event(event_type, extra=""):
    """Appends a timestamped event row to the CSV log."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "event_type", "notes"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), event_type, extra])
    print(f"[log] {event_type} @ {datetime.now().strftime('%H:%M:%S')} {extra}")


def main():
    if not os.path.exists(LANDMARK_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {LANDMARK_MODEL_PATH}. Copy shape_predictor_68_face_landmarks.dat "
            "into the models/ folder."
        )
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {CNN_MODEL_PATH}. Train it first with src/train_model.py."
        )

    detector = get_face_detector()
    predictor = get_landmark_predictor(LANDMARK_MODEL_PATH)
    eye_model = load_model(CNN_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    closed_frame_count = 0
    yawn_frame_count = 0
    drowsy_alert_active = False
    last_alarm_time = 0.0
    yawn_alert_active = False

    print("Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        status_text = "No face detected"
        status_color = (200, 200, 200)

        for face in faces:
            shape = predictor(gray, face)
            landmarks = shape_to_np(shape)

            left_eye_pts = landmarks[LEFT_EYE_IDX]
            right_eye_pts = landmarks[RIGHT_EYE_IDX]
            mouth_pts = landmarks[MOUTH_IDX]

            # --- EAR (cross-check signal) ---
            ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0

            # --- CNN eye-state prediction ---
            left_crop = extract_eye_region(gray, left_eye_pts)
            right_crop = extract_eye_region(gray, right_eye_pts)
            eye_open_probs = []
            for crop in (left_crop, right_crop):
                if crop is not None:
                    prob_open = float(eye_model.predict(crop, verbose=0)[0][0])
                    eye_open_probs.append(prob_open)

            avg_open_prob = np.mean(eye_open_probs) if eye_open_probs else 1.0
            eyes_closed = avg_open_prob < EYE_CLOSED_PROB_THRESHOLD

            # --- MAR (yawn signal) ---
            mar = mouth_aspect_ratio(mouth_pts)
            is_yawning = mar > MAR_YAWN_THRESHOLD

            # --- Drowsy (eyes) state machine ---
            if eyes_closed:
                closed_frame_count += 1
                if closed_frame_count >= CONSEC_CLOSED_FRAMES:
                    now = time.time()
                    if not drowsy_alert_active:
                        drowsy_alert_active = True
                        log_event("DROWSY_START", f"eye_open_prob={avg_open_prob:.2f} ear={ear:.2f}")
                        threading.Thread(target=play_alarm, daemon=True).start()
                        last_alarm_time = now
                    elif now - last_alarm_time >= ALARM_REPEAT_SECONDS:
                        threading.Thread(target=play_alarm, daemon=True).start()
                        last_alarm_time = now
            else:
                if drowsy_alert_active:
                    log_event("DROWSY_END")
                closed_frame_count = 0
                drowsy_alert_active = False

            # --- Yawn state machine ---
            if is_yawning:
                yawn_frame_count += 1
                if yawn_frame_count >= CONSEC_YAWN_FRAMES and not yawn_alert_active:
                    yawn_alert_active = True
                    log_event("YAWN_START", f"mar={mar:.2f}")
            else:
                if yawn_alert_active:
                    log_event("YAWN_END")
                yawn_frame_count = 0
                yawn_alert_active = False

            # --- Overlay ---
            for pts in (left_eye_pts, right_eye_pts):
                hull = cv2.convexHull(pts)
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
            mouth_hull = cv2.convexHull(mouth_pts)
            cv2.drawContours(frame, [mouth_hull], -1, (255, 200, 0), 1)

            if drowsy_alert_active:
                status_text, status_color = "DROWSINESS ALERT!", (0, 0, 255)
            elif yawn_alert_active:
                status_text, status_color = "YAWN DETECTED", (0, 165, 255)
            else:
                status_text, status_color = "Awake", (0, 200, 0)

            cv2.putText(frame, f"Eye-open prob: {avg_open_prob:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.imshow("Drowsiness Detection (CNN + Yawn)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
