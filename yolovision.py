import mediapipe as mp
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import PlayerTracker
from collections import deque

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO("yolov8l.pt")
model.to(device)
tracker = PlayerTracker()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# EMA smoothing factor (higher = more responsive, lower = smoother)
ALPHA = 0.1

# Store previous smoothed angles
smoothed_angles = {
    'left_elbow': None,
    'left_wrist': None,
    'left_knee': None,
    'left_ankle': None,
    'right_elbow': None,
    'right_wrist': None,
    'right_knee': None,
    'right_ankle': None
}

#calculates the angle between three points a, b, c
# a = left shoulder, b = left elbow, c = left wrist
# left shoulder = [x1,y1,z1], left elbow = [x2,y2,z2], left wrist = [x3,y3,z3]
# ba = a - b
# ba = [x1 - x2, y1 - y2, z1 - z2]
# bc = c - b
# bc = [x3 - x2, y3 - y2, z3 - z2]
# cosine_angle = (ba (dot) bc) / (||ba|| * ||bc||)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

#uses exponential smoothing by storing the average of previous values
def smooth_angle(name, new_angle):
    prev = smoothed_angles[name]
    if prev is None:
        smoothed_angles[name] = new_angle
    else:
        smoothed_angles[name] = ALPHA * new_angle + (1 - ALPHA) * prev
    return smoothed_angles[name]

cap = cv2.VideoCapture(0) 
#begins pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            #grabs coordinates in 3D space (x,y,z)
            def get_coords(idx):
                return [lm[idx].x, lm[idx].y, lm[idx].z]

            # all left ligaments
            left_shoulder, left_elbow, left_wrist = get_coords(11), get_coords(13), get_coords(15)
            left_hip, left_knee, left_ankle = get_coords(23), get_coords(25), get_coords(27)
            left_heel, left_foot = get_coords(29), get_coords(31)

            # all right ligaments
            right_shoulder, right_elbow, right_wrist = get_coords(12), get_coords(14), get_coords(16)
            right_hip, right_knee, right_ankle = get_coords(24), get_coords(26), get_coords(28)
            right_heel, right_foot = get_coords(30), get_coords(32)

            # run calculate angle function for each joint
            angles = {
                'left_elbow': calculate_angle(left_shoulder, left_elbow, left_wrist),
                'left_wrist': calculate_angle(left_elbow, left_wrist, left_foot),
                'left_knee': calculate_angle(left_hip, left_knee, left_ankle),
                'left_ankle': calculate_angle(left_knee, left_ankle, left_heel),

                'right_elbow': calculate_angle(right_shoulder, right_elbow, right_wrist),
                'right_wrist': calculate_angle(right_elbow, right_wrist, right_foot),
                'right_knee': calculate_angle(right_hip, right_knee, right_ankle),
                'right_ankle': calculate_angle(right_knee, right_ankle, right_heel)
            }

            # # now apply smoothing to angles continuously
            # for key in angles:
            #     angles[key] = smooth_angle(key, angles[key])

            #function that draws the text for each joiny
            def draw_text(label, coords, value, color):
                pos = tuple((np.array(coords[:2]) * [w, h]).astype(int))
                cv2.putText(image, f'{label}: {int(value)}°', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # draw angles for left and right joints
            draw_text('L-Elbow', left_elbow, angles['left_elbow'], (0, 255, 0))
            draw_text('L-Wrist', left_wrist, angles['left_wrist'], (255, 255, 0))
            draw_text('L-Knee', left_knee, angles['left_knee'], (255, 0, 0))
            draw_text('L-Ankle', left_ankle, angles['left_ankle'], (0, 0, 255))

            draw_text('R-Elbow', right_elbow, angles['right_elbow'], (0, 255, 0))
            draw_text('R-Wrist', right_wrist, angles['right_wrist'], (255, 255, 0))
            draw_text('R-Knee', right_knee, angles['right_knee'], (255, 0, 0))
            draw_text('R-Ankle', right_ankle, angles['right_ankle'], (0, 0, 255))

            # Draw full skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Angles (Left + Right, Smoothed)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
