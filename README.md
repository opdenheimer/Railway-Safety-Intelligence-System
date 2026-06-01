# AI-Powered Railway Safety Intelligence System (MVP)

A computer vision prototype designed to monitor railway tracks, detect unauthorized intrusions (people, animals, debris) into danger zones, and log incidents in real-time. 

This Minimal Viable Product (MVP) focuses on edge-computing principles using 100% offline detection methods, requiring no external AI model downloads.

## 🌟 Features

- **Real-Time Intrusion Detection:** Automatically detects moving objects on the tracks.
- **Configurable Danger Zone:** An adjustable polygon (Region of Interest) overlay allows users to define the specific track area.
- **Size-Based Classification:** Roughly categorizes objects (person/animal, debris) based on contour area.
- **Automated Logging:** Saves all intrusion incidents (timestamp, class, track ID, confidence) to a local SQLite database.
- **Interactive Dashboard:** Built with Streamlit, featuring a live video feed, alert history, and performance controls.
- **Zero-Download AI:** Uses OpenCV's robust background subtraction algorithms, completely bypassing restrictive corporate firewalls that block model weights (like YOLO).

## 🛠️ Technology Stack

- **Language:** Python 3
- **Computer Vision:** OpenCV (`cv2`)
- **Dashboard Frontend:** Streamlit
- **Geometry Logic:** Shapely (Polygon intersection)
- **Database:** SQLite3
- **Data Manipulation:** Pandas

## 🧠 Core Computer Vision Concepts Demonstrated

1. **Background Subtraction (MOG2):** The system continuously learns the static background of the video feed (the empty tracks). It isolates anything that moves (the foreground).
2. **Morphological Operations:** Applies Opening, Closing, and Dilation to the foreground mask to remove noise, shadows, and fill in holes in moving objects.
3. **Contour Analysis:** Finds the boundaries of moving objects and calculates their area to filter out tiny artifacts and roughly classify the object size.
4. **Polygon Intersection:** Uses `Shapely` to determine if the bottom-center point of a detected moving object has crossed into the user-defined polygonal Danger Zone.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/railway-safety-mvp.git
cd railway-safety-mvp
```

### 2. Set up the Python Environment
It is highly recommended to use a virtual environment.
```bash
python -m venv CVvenv

# Windows
.\CVvenv\Scripts\activate

# Mac/Linux
source CVvenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Provide a Video
Place a sample video file named `sample_video.mp4` directly in the project root directory, or use the "Upload Video" feature in the dashboard. *(Note: 4K 60fps videos are supported but will automatically have frames skipped/resized for performance).*

### 5. Run the Application
```bash
streamlit run app.py
```
The dashboard will automatically open in your default web browser at `http://localhost:8501`.

## 🎮 How to Use the Dashboard

1. **Select Video Source:** Choose between a local sample video, uploading a new video, or using your webcam.
2. **Adjust the Danger Zone:** Use the sliders in the sidebar to shape the red trapezoid so it perfectly covers the railway tracks in your specific camera angle.
3. **Performance Tuning:** If the video feed is lagging, lower the processing resolution or increase the frame-skip counter in the sidebar.
4. **Start Detection:** Click the `▶️ Start Detection` button. 
5. **Monitor:** Green boxes indicate safe moving objects. Red boxes indicate an active intrusion into the Danger Zone, which will be logged in the "Recent Incidents" table on the right.

## 🔮 Future Enhancements (Post-MVP)
- Integration with YOLOv8/YOLO11 for high-accuracy, multi-class object detection.
- Implementation of DeepSORT/ByteTrack for persistent unique object tracking IDs across occlusions.
- Migration to a FastAPI backend and PostgreSQL database for enterprise scalability.
- Cloud deployment via Docker.
