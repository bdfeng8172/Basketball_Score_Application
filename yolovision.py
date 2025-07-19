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

        # Run detection
        results = model.predict(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0].cpu().item())
            label = model.names[cls_id]

            # Only track people
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0].cpu().item())
                detections.append([x1, y1, x2, y2, score])
        

        detections = np.array(detections, dtype=np.float32)

        img_info = {
            'height': frame.shape[0],
            'width': frame.shape[1]
        }

        img_size = (frame.shape[0], frame.shape[1])

        tracked_players = tracker.update(detections, img_info, img_size)

        current_detection_count = len(detections)

        if current_detection_count != prev_detection_count:
            print(f"Detections: {current_detection_count}, Tracked: {tracked_players}")
            prev_detection_count = current_detection_count

        for tid, x1, y1, x2, y2 in tracked_players:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Player {tid}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Stream the frame
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
    cap.release()
    cv2.destroyAllWindows()
