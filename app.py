"""
app.py — Streamlit dashboard for the Drowsiness Detection system.

Run locally with:
    streamlit run app.py

Note: this reads the LOCAL machine's webcam via OpenCV, so it must be
run on your own computer (not usable as-is on Streamlit Community Cloud,
which has no webcam access — fine for local demos and portfolio videos).
"""

import os
import sys
import time
import threading

import cv2
import imutils
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from landmarks import (  # noqa: E402
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
from detect import log_event, LOG_PATH, play_alarm, ALARM_REPEAT_SECONDS  # noqa: E402
LANDMARK_MODEL_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
CNN_MODEL_PATH = os.path.join("models", "eye_state_cnn.keras")

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("😴 Driver Drowsiness Detection — CNN + Yawn Detection")
st.caption(
    "Hybrid detection: a trained CNN classifies eye state (Open/Closed) "
    "per frame, combined with Mouth Aspect Ratio for yawn detection. "
    "Every drowsy/yawn event is timestamped and logged."
)

with st.sidebar:
    st.header("⚙️ Settings")
    closed_prob_threshold = st.slider("Eye-closed probability threshold", 0.1, 0.9, 0.5, 0.05)
    consec_closed_frames = st.slider("Consecutive closed frames to alert", 5, 40, 20)
    mar_threshold = st.slider("Yawn (MAR) threshold", 0.3, 1.0, 0.6, 0.05)
    consec_yawn_frames = st.slider("Consecutive yawn frames to alert", 5, 30, 15)
    run_detection = st.checkbox("▶ Start Detection")

col_video, col_status = st.columns([2, 1])
frame_placeholder = col_video.empty()
status_placeholder = col_status.empty()

log_header = st.container()
log_header.subheader("📋 Session Event Log")
log_placeholder = st.empty()


@st.cache_resource
def load_models():
    detector = get_face_detector()
    predictor = get_landmark_predictor(LANDMARK_MODEL_PATH)
    eye_model = load_model(CNN_MODEL_PATH)
    return detector, predictor, eye_model


def render_log():
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        log_placeholder.dataframe(df.tail(20), use_container_width=True)
    else:
        log_placeholder.info("No events logged yet this session.")


if run_detection:
    if not os.path.exists(LANDMARK_MODEL_PATH):
        st.error(f"Missing {LANDMARK_MODEL_PATH}. Add the dlib landmark model to models/.")
    elif not os.path.exists(CNN_MODEL_PATH):
        st.error(f"Missing {CNN_MODEL_PATH}. Train it first: python src/train_model.py")
    else:
        detector, predictor, eye_model = load_models()
        cap = cv2.VideoCapture(0)

        closed_frame_count = 0
        yawn_frame_count = 0
        drowsy_alert_active = False
        yawn_alert_active = False
        last_alarm_time = 0.0

        while run_detection and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read from webcam.")
                break

            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            status_text, status_kind = "No face detected", "info"

            for face in faces:
                shape = predictor(gray, face)
                landmarks = shape_to_np(shape)

                left_eye_pts = landmarks[LEFT_EYE_IDX]
                right_eye_pts = landmarks[RIGHT_EYE_IDX]
                mouth_pts = landmarks[MOUTH_IDX]

                ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0

                probs = []
                for pts in (left_eye_pts, right_eye_pts):
                    crop = extract_eye_region(gray, pts)
                    if crop is not None:
                        probs.append(float(eye_model.predict(crop, verbose=0)[0][0]))
                avg_open_prob = np.mean(probs) if probs else 1.0
                eyes_closed = avg_open_prob < closed_prob_threshold

                mar = mouth_aspect_ratio(mouth_pts)
                is_yawning = mar > mar_threshold

                if eyes_closed:
                    closed_frame_count += 1
                    if closed_frame_count >= consec_closed_frames:
                        now = time.time()
                        if not drowsy_alert_active:
                            drowsy_alert_active = True
                            log_event("DROWSY_START", f"eye_open_prob={avg_open_prob:.2f}")
                            threading.Thread(target=play_alarm, daemon=True).start()
                            last_alarm_time = now
                        elif now - last_alarm_time >= ALARM_REPEAT_SECONDS:
                            threading.Thread(target=play_alarm, daemon=True).start()
                            last_alarm_time = now
                else:
                    if drowsy_alert_active:
                        log_event("DROWSY_END")
                    closed_frame_count, drowsy_alert_active = 0, False

                if is_yawning:
                    yawn_frame_count += 1
                    if yawn_frame_count >= consec_yawn_frames and not yawn_alert_active:
                        yawn_alert_active = True
                        log_event("YAWN_START", f"mar={mar:.2f}")
                else:
                    if yawn_alert_active:
                        log_event("YAWN_END")
                    yawn_frame_count, yawn_alert_active = 0, False

                for pts in (left_eye_pts, right_eye_pts):
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth_pts)], -1, (255, 200, 0), 1)

                if drowsy_alert_active:
                    status_text, status_kind = "🔴 DROWSINESS ALERT!", "error"
                elif yawn_alert_active:
                    status_text, status_kind = "🟠 YAWN DETECTED", "warning"
                else:
                    status_text, status_kind = "🟢 Awake", "success"

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            getattr(status_placeholder, status_kind)(status_text)
            render_log()
            time.sleep(0.03)

        cap.release()
else:
    frame_placeholder.info("Tick **Start Detection** in the sidebar to begin.")
    render_log()
