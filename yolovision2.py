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

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Rolling average buffer (for smoothing angles)
SMOOTH_WINDOW = 7
elbow_angles = deque(maxlen=SMOOTH_WINDOW)
wrist_angles = deque(maxlen=SMOOTH_WINDOW)
knee_angles = deque(maxlen=SMOOTH_WINDOW)
ankle_angles = deque(maxlen=SMOOTH_WINDOW)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def smooth_angle(angle_list, new_angle):
    angle_list.append(new_angle)
    return np.mean(angle_list)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_detection_count = 0

def detect_and_stream():
    global prev_detection_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model.predict(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0].cpu().item())
            label = model.names[cls_id]
            if label in ["person", "sports ball"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0].cpu().item())
                w = x2 - x1
                h = y2 - y1
                detections.append([x1, y1, w, h, score, cls_id])

        if len(detections) > 0:
            detections = np.array(detections, dtype=np.float32)
            dets_for_tracker = detections[:, :5]
            cls_ids = detections[:, 5].astype(int)
        else:
            dets_for_tracker = np.empty((0, 5), dtype=np.float32)
            cls_ids = np.empty((0,), dtype=int)

        img_info = {'height': frame.shape[0], 'width': frame.shape[1]}
        img_size = (frame.shape[1], frame.shape[0])

        tracked_objects = tracker.update(dets_for_tracker, img_info, img_size, cls_ids)

        current_detection_count = len(detections)
        if current_detection_count != prev_detection_count:
            print(f"Detections: {current_detection_count}, Tracked: {tracked_objects}")
            prev_detection_count = current_detection_count

        # Draw tracked objects
        for tid, x1, y1, x2, y2, cls_id in tracked_objects:
            if cls_id == 0:
                color = (255, 0, 0)
                label = "Person"
            elif cls_id == 32:
                color = (0, 255, 255)
                label = "Ball"
            else:
                color = (0, 0, 255)
                label = f"Class {cls_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {tid}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Run Mediapipe Pose
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = pose_results.pose_landmarks.landmark

            def get_coords(idx):
                return [lm[idx].x, lm[idx].y, lm[idx].z]

            # Left-side landmarks
            shoulder, elbow, wrist = get_coords(11), get_coords(13), get_coords(15)
            hip, knee, ankle = get_coords(23), get_coords(25), get_coords(27)
            heel, index = get_coords(29), get_coords(19)

            # Calculate and smooth angles
            elbow_angle = smooth_angle(elbow_angles, calculate_angle(shoulder, elbow, wrist))
            wrist_angle = smooth_angle(wrist_angles, calculate_angle(elbow, wrist, index))
            knee_angle = smooth_angle(knee_angles, calculate_angle(hip, knee, ankle))
            ankle_angle = smooth_angle(ankle_angles, calculate_angle(knee, ankle, heel))

            # Display angles on video frame
            h, w, _ = frame.shape
            def draw_text(text, coords, color):
                pos = tuple(np.multiply(coords[:2], [w, h]).astype(int))
                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            draw_text(f'Elbow: {int(elbow_angle)}°', elbow, (0,255,0))
            draw_text(f'Wrist: {int(wrist_angle)}°', wrist, (255,255,0))
            draw_text(f'Knee: {int(knee_angle)}°', knee, (255,0,0))
            draw_text(f'Ankle: {int(ankle_angle)}°', ankle, (0,0,255))

        # Encode frame for MJPEG streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
