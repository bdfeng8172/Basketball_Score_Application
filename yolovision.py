import mediapipe as mp
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import PlayerTracker

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO("yolov8l.pt")
model.to(device)
tracker = PlayerTracker()

print(model.device) 

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

# Load webcam
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

        # have a rgb copy for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO
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

        img_info = {
            'height': frame.shape[0],
            'width': frame.shape[1],
        }
        img_size = (frame.shape[1], frame.shape[0])

        tracked_objects = tracker.update(dets_for_tracker, img_info, img_size, cls_ids)

        current_detection_count = len(detections)
        if current_detection_count != prev_detection_count:
            print(f"Detections: {current_detection_count}, Tracked: {tracked_objects}")
            prev_detection_count = current_detection_count

        # Label and draw tracked objects
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

        # run Mediapipe Pose
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            # draw pose skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # calculate center of shoulders and hips
            lm = pose_results.pose_landmarks.landmark

            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            cx = int(frame.shape[1] * (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4)
            cy = int(frame.shape[0] * (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4)

            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, "Pose Center", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
