"""
landmarks.py
Face landmark detection utilities: dlib 68-point face landmarks,
Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and eye-region
cropping used to feed the CNN eye-state classifier.
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# 68-point landmark index ranges (standard iBUG 300-W scheme)
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))
MOUTH_IDX = list(range(48, 68))

EYE_CNN_INPUT_SIZE = (24, 24)  # width, height fed into the CNN


def get_face_detector():
    """Returns dlib's HOG-based frontal face detector."""
    return dlib.get_frontal_face_detector()


def get_landmark_predictor(model_path):
    """Loads dlib's 68-point facial landmark predictor."""
    return dlib.shape_predictor(model_path)


def shape_to_np(shape, dtype="int"):
    """Converts a dlib shape object to a (68, 2) numpy array."""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye_points):
    """
    Classic EAR formula (Soukupova & Cech, 2016).
    eye_points: 6 (x, y) coordinates for one eye.
    Used here as a fast, interpretable cross-check alongside the CNN.
    """
    a = dist.euclidean(eye_points[1], eye_points[5])
    b = dist.euclidean(eye_points[2], eye_points[4])
    c = dist.euclidean(eye_points[0], eye_points[3])
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(mouth_points):
    """
    MAR formula for yawn detection using the inner mouth landmarks
    (indices 60-67 within the 68-point set, i.e. mouth_points[12:20]).
    """
    a = dist.euclidean(mouth_points[13], mouth_points[19])
    b = dist.euclidean(mouth_points[14], mouth_points[18])
    c = dist.euclidean(mouth_points[15], mouth_points[17])
    d = dist.euclidean(mouth_points[12], mouth_points[16])
    return (a + b + c) / (3.0 * d)


def extract_eye_region(gray_frame, eye_points, padding=5):
    """
    Crops a tight bounding box around one eye from a grayscale frame,
    resizes it to the CNN's expected input size, and normalizes it.
    Returns a (24, 24, 1) float32 array ready for model.predict(),
    or None if the crop is degenerate (out of frame bounds).
    """
    x_coords = eye_points[:, 0]
    y_coords = eye_points[:, 1]
    x1, x2 = max(0, x_coords.min() - padding), x_coords.max() + padding
    y1, y2 = max(0, y_coords.min() - padding), y_coords.max() + padding

    eye_crop = gray_frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None

    eye_resized = cv2.resize(eye_crop, EYE_CNN_INPUT_SIZE)
    eye_normalized = eye_resized.astype("float32") / 255.0
    return eye_normalized.reshape(1, EYE_CNN_INPUT_SIZE[1], EYE_CNN_INPUT_SIZE[0], 1)
