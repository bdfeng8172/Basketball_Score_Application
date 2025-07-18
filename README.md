# 🏀 Basketball Score App — YOLOv8 + ByteTrack + Flask

This project is a **real-time basketball player tracking** app.  
It uses **YOLOv8** for object detection and **ByteTrack** for tracking players across frames.  
A simple **Flask server** streams the live video feed with tracked players in your browser.

---

## 📂 Project Structure

Basketball_Score_App/
├── app.py # Flask server entry point
├── yolovision.py # YOLO detection + streaming
├── tracker.py # ByteTrack wrapper
├── ByteTrack/ # Local copy of ByteTrack repo
├── templates/
│ └── index.html # Web UI
├── requirements.txt # Python dependencies
├── yolov8.pt # pre trained models of different sizes5
└── README.md # This file

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

## TO INSTALL REQUIRED PACKAGES 

pip install -r requirements.txt



## Troubleshooting

ModuleNotFoundError — Did you activate your virtual environment?

C++ compiler error — Install Visual C++ Build Tools.

Webcam not showing — Check your webcam permissions.

CUDA not detected — Confirm you installed the matching torch + torchvision version.