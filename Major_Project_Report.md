# Major Project Report: AI Road Pothole Detector

*(Note: Copy this content into Microsoft Word or Google Docs to add your college's specific Title Page, Certificate, Declaration, and Acknowledgement pages before saving as a PDF.)*

---

## 1. ABSTRACT
Poor road infrastructure, particularly potholes, is a significant cause of traffic congestion, vehicle damage, and severe accidents globally. Traditional methods of monitoring road conditions rely heavily on manual surveys or sporadic citizen complaints, which are inefficient, time-consuming, and prone to delays. This project proposes an automated "AI Road Pothole Detector," a web-based application designed to stream real-time camera feeds or process uploaded media (images and videos) to detect potholes using advanced Computer Vision techniques. 

The system utilizes the state-of-the-art Ultralytics YOLOv8 (You Only Look Once) object detection model, exported in ONNX format for rapid inference without requiring heavy computational resources. When a pothole is detected, the application leverages the Geopy library and OpenStreetMap data to capture the user's exact GPS coordinates and reverse-geocode them into a physical address. Finally, the system automatically drafts a comprehensive complaint payload containing the number of potholes, mapped location, spatial markers, and photographic evidence, enabling citizens to easily lodge actionable reports to municipal authorities (e.g., GHMC). The system includes smart reporting logic to prevent false or blank complaints when no potholes are detected, ensuring high-quality data submission.

## 2. INTRODUCTION
### 2.1 Problem Statement
Urban regions struggle to maintain road infrastructure due to the sheer volume of road networks. Identifying potholes requires massive manpower and budget. By the time a citizen traditionally submits a written complaint and the government processes it, the pothole may have already caused severe vehicular accidents. There is a critical need for an automated, AI-driven, and citizen-accessible platform to report road hazards instantaneously.

### 2.2 Objectives
* To develop an accurate pothole detection model using Deep Learning (YOLOv8).
* To create a responsive web utility accessible via smartphones and laptops.
* To automatically extract and append accurate geolocation data (Latitude, Longitude, and Street Address) to detected hazards.
* To bridge the gap between citizens and government bodies by automating the formal complaint generation process.
* To maintain a persistent local history of user-generated reports.

## 3. EXISTING SYSTEM VS. PROPOSED SYSTEM
### 3.1 Existing System
Currently, municipal corporations rely on either manual vehicular inspections or outdated citizen portal applications where a user must manually type their address, describe the issue, and manually attach photos. 
**Drawbacks of Existing Systems:**
* High manual effort for the user.
* Lack of standardized visual evidence (e.g., no bounding boxes indicating the hazard).
* Inaccurate location reporting by citizens.
* High latency between complaint creation and resolution.

### 3.2 Proposed System
The AI Pothole Detector automates the recognition and reporting pipeline. The user simply points their camera at the road; the AI highlights the potholes, the GPS fetches the location, and a one-click digital complaint is generated.
**Advantages:**
* Extremely fast inference using the ONNX framework.
* Progressive asynchronous loading reduces app latency.
* "Zero-Typing" complaint formulation.
* Prevents spam by disabling report generation if 0 potholes are detected.

## 4. SYSTEM REQUIREMENTS
### 4.1 Hardware Requirements
* **Processor:** Minimum Intel Core i3 / AMD Ryzen 3 (or equivalent smartphone processor).
* **RAM:** 4 GB RAM (8 GB Recommended).
* **Peripherals:** Integrated Webcam or Smartphone Camera.

### 4.2 Software Requirements
* **Operating System:** Windows / Linux / macOS.
* **Programming Languages:** Python 3.8+, JavaScript, HTML5, CSS3.
* **Frameworks & Libraries:** 
  * Flask (Backend Server)
  * Ultralytics YOLOv8 (AI Model)
  * OpenCV (`cv2`) (Image Processing)
  * Geopy (Geolocation mapping)
* **Browser:** Any modern web browser (Chrome, Safari, Edge) with Geolocation and Camera hardware permissions enabled.

## 5. SYSTEM ARCHITECTURE & METHODOLOGY
### 5.1 Architecture Overview
The application follows a Client-Server architecture:
1. **Presentation Layer (Frontend):** Built with HTML/CSS/JS. Handles the camera streams, drag-and-drop media uploads, optimistic UI rendering, and user interactions.
2. **Application Layer (Backend):** A Python Web Server (Flask) that routes REST API requests and manages file saving.
3. **AI / Processing Layer:** The YOLOv8 ONNX model parses `.jpg` frames or `.webm/.mp4` video chunks, applies bounding boxes to recognized potholes, and counts them.
4. **External Services:** The backend pushes coordinates to the Nominatim (OpenStreetMap) API to return a formatted street-level address string.

### 5.2 YOLOv8 Model
YOLOv8 is a single-stage object detection model known for its speed and accuracy. The model was trained on a dataset of road anomalies and exported to `.onnx` (Open Neural Network Exchange). This allows the application to drop heavy PyTorch dependencies and run lightweight, CPU-friendly inference algorithms, making the application heavily scalable.

### 5.3 Flow of Execution
1. User provides media (Live Camera / File Upload).
2. Media sent via POST request to `/api/detect_single`.
3. Parallel execution: 
   - YOLOv8 calculates bounding boxes and total hazard count.
   - Geopy fetches device coordinates via navigator API and returns the physical address.
4. Results combined and displayed on UI.
5. If Hazard Count > 0: Options to route to `/complaint` are unlocked.
6. Payload packaged as a dictionary and finalized into a standardized JSON endpoint for governmental use.

## 6. IMPLEMENTATION & MODULES
### 6.1 User Interface Module
Features a robust "App-Shell" design. It contains a real-time `navigator.mediaDevices.getUserMedia` hook for establishing secure camera feeds (front/rear toggling).

### 6.2 Computer Vision Module
Using OpenCV, incoming video feeds are segmented into individual frames (e.g., 1 frame every 30 frames for optimization). These frames are passed through `model(path, conf=0.25)`. A confidence threshold of 25% prevents false positives while capturing genuine road defects.

### 6.3 Routing & History Module
The application uses browser `localStorage` and `sessionStorage` to maintain state. Users can navigate freely between the Home screen, the Results abstract, and the History log without losing previous detection data. 

## 7. CONCLUSION
The AI Road Pothole Detector successfully demonstrates how modern web applications and Artificial Intelligence can intersect to solve critical civic infrastructure issues. By automating the hardest parts of hazard reporting—namely visual identification and geolocation—the system drastically lowers the friction for citizens to help maintain their cities. It provides municipal bodies with structured, highly accurate, and standardized data (including verifiable images and coordinates), which can drastically reduce the time needed to dispatch road maintenance teams.

## 8. FUTURE SCOPE
* **Government Integration:** Directly bridging the payload generator to a live Municipal Corporation database (like GHMC) via SOAP/REST API.
* **Edge Deployment:** Deploying the model onto dashcams inside public transport (e.g., city buses or garbage trucks) for passive, continuous 24/7 scanning of city streets.
* **Severity Grading:** Enhancing the AI to classify potholes based on depth and size (e.g., Low, Medium, Critical hazard) to help governments prioritize emergency repairs.
* **Crowdsourced Heatmaps:** Building a global dashboard that visualizes all user-submitted potholes on a live city map.

---

## 9. REFERENCES
1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. 
2. Flask Documentation. (2024). Web application framework for Python.
3. OpenStreetMap contributors. (2024). Nominatim API for Geocoding.
4. Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
