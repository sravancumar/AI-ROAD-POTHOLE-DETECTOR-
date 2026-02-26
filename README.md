# Pothole Detector (YOLO ONNX Web App)

A Flask web application that uses a YOLO model exported to ONNX (`pothole_guard.onnx`) to detect road potholes from camera, video, or uploaded images. The app works on desktop and mobile, shows pothole counts and location on a map, keeps a local history, and generates a formatted report for complaints.

## Features

- **ONNX model inference** using Ultralytics YOLO API (`pothole_guard.onnx`)
- **Image & video input**
  - Live camera capture (HTTPS / Hugging Face / localhost)
  - Video recording (WebM) from the camera
  - File upload for images and videos
- **Result view**
  - Pothole count
  - Address and coordinates
  - Embedded Google Maps view
  - Detected images grid
- **History & reporting**
  - Local history in browser `localStorage`
  - “Official Report” page with formatted text
  - Options for email / WhatsApp / GHMC integration

## Tech stack & versions

- Python 3.10.x
- Flask 3.0.2
- Ultralytics 8.4.14
- Torch 2.10.0 (CPU)
- ONNX Runtime 1.20.1
- NumPy 2.2.6
- OpenCV (headless) 4.13.0.92
- Geopy, Pillow, ReportLab

## Project structure

- `app.py` – Flask app, ONNX model loading, routes
- `pothole_guard.onnx` – YOLO model in ONNX format
- `templates/` – HTML templates (`index`, `result`, `last_result`, `complaint`, `history`)
- `static/`
  - `style.css` – mobile‑first UI styling
  - `results/` – generated detection images (created at runtime)
- `scripts/` – optional GHMC helper script
- `requirements.txt` – Python dependencies
- `runtime.txt` – Python version hint for deployments
- `Dockerfile` – container setup

## Running locally (without Docker)

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   pip install -r requirements.txt
   
