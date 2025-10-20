import mediapipe as mp
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import PlayerTracker
from collections import deque
from form_analysis.basketballFAmod import BasketballFormAnalysis

# initialize form analyzer
form_analyzer = BasketballFormAnalysis()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


model = YOLO("yolov8l.pt")
model.to(device)
tracker = PlayerTracker()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# smoothing factor (higher = more responsive, lower = smoother)
ALPHA = 0.1

# store previous smoothed angles
smoothed_angles = {
    'left_elbow': None, 'left_wrist': None, 'left_knee': None, 'left_ankle': None,
    'right_elbow': None, 'right_wrist': None, 'right_knee': None, 'right_ankle': None,
    'left_shoulder': None, 'right_shoulder': None
}


#  specific 2D angle calculation function
def calculate_angle_2d(a, b, c):

    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])  # Use only x and y
    ba = a - b
    bc = c - b

    # Calculate cosine using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clamp values to prevent floating-point issues (e.g., >1.0 or <-1.0)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Convert from radians to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# function that utilizes exponential moving average for smoothing
def smooth_angle(name, new_angle):
    prev = smoothed_angles[name]
    if prev is None:
        smoothed_angles[name] = new_angle
    else:
        smoothed_angles[name] = ALPHA * new_angle + (1 - ALPHA) * prev
    return smoothed_angles[name]

video = "basketball_shot.mp4"  # Replace with 0 for webcam input
cap = cv2.VideoCapture(video)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    
    form_analyzer = BasketballFormAnalysis()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # converts to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # converts back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # function to get normalized coordinates
            def get_coords(idx):
                return [lm[idx].x, lm[idx].y, lm[idx].z]

            # left side joints
            left_shoulder, left_elbow, left_wrist = get_coords(11), get_coords(13), get_coords(15)
            left_hip, left_knee, left_ankle = get_coords(23), get_coords(25), get_coords(27)
            left_heel, left_foot = get_coords(29), get_coords(31)

            # right side joints
            right_shoulder, right_elbow, right_wrist = get_coords(12), get_coords(14), get_coords(16)
            right_hip, right_knee, right_ankle = get_coords(24), get_coords(26), get_coords(28)
            right_heel, right_foot = get_coords(30), get_coords(32)

            # CALCULATE 2D ANGLES 
            angles = {
                'left_elbow': calculate_angle_2d(left_shoulder, left_elbow, left_wrist),
                'left_wrist': calculate_angle_2d(left_elbow, left_wrist, left_foot),
                'left_knee': calculate_angle_2d(left_hip, left_knee, left_ankle),
                'left_ankle': calculate_angle_2d(left_knee, left_ankle, left_heel),
                'left_shoulder': calculate_angle_2d(left_elbow, left_shoulder, left_hip),

                'right_elbow': calculate_angle_2d(right_shoulder, right_elbow, right_wrist),
                'right_wrist': calculate_angle_2d(right_elbow, right_wrist, right_foot),
                'right_knee': calculate_angle_2d(right_hip, right_knee, right_ankle),
                'right_ankle': calculate_angle_2d(right_knee, right_ankle, right_heel),
                'right_shoulder': calculate_angle_2d(right_elbow, right_shoulder, right_hip),
            }

            # apply smoothing
            for key in angles:
                angles[key] = smooth_angle(key, angles[key])
            # perform form analysis
            for position in ["position_1", "position_2", "position_3"]:
                overall_status, joint_statuses = form_analyzer.evaluate_position(smoothed_angles, position)

            # function to draw text
            def draw_text(label, coords, value, color):
                pos = tuple((np.array(coords[:2]) * [w, h]).astype(int))
                cv2.putText(image, f'{label}: {int(value)}°', pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Draw left-side angles
            draw_text('L-Elbow', left_elbow, angles['left_elbow'], (0, 255, 0))
            draw_text('L-Wrist', left_wrist, angles['left_wrist'], (255, 255, 0))
            draw_text('L-Knee', left_knee, angles['left_knee'], (255, 0, 0))
            draw_text('L-Ankle', left_ankle, angles['left_ankle'], (0, 0, 255))
            draw_text('L-Shoulder', left_shoulder, angles['left_shoulder'], (255, 0, 255))

            # Draw right-side angles
            draw_text('R-Elbow', right_elbow, angles['right_elbow'], (0, 255, 0))
            draw_text('R-Wrist', right_wrist, angles['right_wrist'], (255, 255, 0))
            draw_text('R-Knee', right_knee, angles['right_knee'], (255, 0, 0))
            draw_text('R-Ankle', right_ankle, angles['right_ankle'], (0, 0, 255))
            draw_text('R-Shoulder', right_shoulder, angles['right_shoulder'], (255, 0, 255))

            # Draw full skeleton
            # After obtaining joint_feedback from form_analyzer
            overall_status, joint_feedback = form_analyzer.evaluate_position(smoothed_angles, "position_1")

# Draw the pose skeleton with color-coded joints
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = image.shape

    # Helper function to convert Mediapipe coords to pixels
                def to_pixel_coords(landmark):
                    return int(landmark.x * w), int(landmark.y * h)

    # Define pairs of landmarks (based on Mediapipe’s POSE_CONNECTIONS)
                connections = [
                    (11, 13), (13, 15),   # Left arm
                    (12, 14), (14, 16),   # Right arm
                    (23, 25), (25, 27),   # Left leg
                    (24, 26), (26, 28),   # Right leg
                    (11, 12), (23, 24),   # Shoulders & hips
                    (11, 23), (12, 24)    # Torso
                ]

                # Mapping landmark pairs to logical joint names for color lookup
                connection_to_joint = {
                    (11, 13): "left_shoulder",
                    (13, 15): "left_elbow",
                    (12, 14): "right_shoulder",
                    (14, 16): "right_elbow",
                    (23, 25): "left_hip",
                    (25, 27): "left_knee",
                    (24, 26): "right_hip",
                    (26, 28): "right_knee"
                }

                for start_idx, end_idx in connections:
                    start_point = to_pixel_coords(lm[start_idx])
                    end_point = to_pixel_coords(lm[end_idx])
                # Default to gray if joint not in feedback
                    joint_name = connection_to_joint.get((start_idx, end_idx))
                    if joint_name and joint_name in joint_feedback:
                        color = joint_feedback[joint_name]["color"]
                    else:
                        color = (128, 128, 128)

                    # Draw colored skeleton segment
                    cv2.line(image, start_point, end_point, color, 4)

                # Optionally draw circles for each landmark
                for idx, landmark in enumerate(lm):
                    cx, cy = to_pixel_coords(landmark)
                    cv2.circle(image, (cx, cy), 4, (255, 255, 255), -1)

        

        cv2.imshow('Pose Angles (2D, Smoothed)', image)

        # Quit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

