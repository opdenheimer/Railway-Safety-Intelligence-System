import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from tracker import RailwayTracker
from database import init_db, log_incident, get_recent_incidents
import time

st.set_page_config(layout="wide", page_title="Railway Safety MVP")

st.title("🚆 Railway Safety Intelligence System MVP")
st.markdown("Real-Time Track Intrusion and Hazard Detection")

# Initialize database
init_db()

# Session state for controlling the detection loop
if "running" not in st.session_state:
    st.session_state.running = False

@st.cache_resource
def load_tracker():
    return RailwayTracker()

tracker = load_tracker()

# Sidebar for controls
st.sidebar.header("Configuration")
video_source = st.sidebar.radio("Video Source", ["Sample Video", "Upload Video", "Webcam"])

video_path = None
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()  # Close before OpenCV reads it (fixes Windows file lock bug)
        video_path = tfile.name
elif video_source == "Webcam":
    video_path = 0  # OpenCV webcam index
else:
    sample_path = "sample_video.mp4"
    if os.path.exists(sample_path):
        video_path = sample_path
    else:
        st.sidebar.warning(
            "No sample video found. Please place a `sample_video.mp4` "
            "in the project directory, or switch to 'Upload Video'."
        )
        video_path = None

# Zone Adjustment
st.sidebar.header("Danger Zone Settings")
st.sidebar.markdown("Adjust the normalized coordinates of the danger zone polygon (trapezoid).")
z_top_left_x = st.sidebar.slider("Top Left X", 0.0, 1.0, 0.3, key="ztlx")
z_top_right_x = st.sidebar.slider("Top Right X", 0.0, 1.0, 0.7, key="ztrx")
z_top_y = st.sidebar.slider("Top Y", 0.0, 1.0, 0.4, key="zty")
z_bottom_left_x = st.sidebar.slider("Bottom Left X", 0.0, 1.0, 0.0, key="zblx")
z_bottom_right_x = st.sidebar.slider("Bottom Right X", 0.0, 1.0, 1.0, key="zbrx")
z_bottom_y = st.sidebar.slider("Bottom Y", 0.0, 1.0, 1.0, key="zby")

tracker.zone_polygon_normalized = [
    (z_top_left_x, z_top_y), (z_top_right_x, z_top_y),
    (z_bottom_right_x, z_bottom_y), (z_bottom_left_x, z_bottom_y)
]

# Performance settings
st.sidebar.header("Performance")
process_width = st.sidebar.selectbox(
    "Processing Resolution (width)", 
    [640, 960, 1280, 1920], 
    index=1,
    help="Lower = faster processing, Higher = better detection"
)
frame_skip = st.sidebar.slider(
    "Process every N-th frame", 1, 10, 2,
    help="Skip frames to speed up processing. 1 = every frame, 3 = every 3rd frame"
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_button = st.button("▶️ Start Detection", use_container_width=True)
    with btn_col2:
        stop_button = st.button("⏹️ Stop Detection", use_container_width=True)

with col2:
    st.subheader("Recent Incidents")
    incident_placeholder = st.empty()
    alert_count_placeholder = st.empty()

def update_incidents_table():
    rows = get_recent_incidents(10)
    if rows:
        df = pd.DataFrame(rows, columns=["Timestamp", "Type", "Track ID", "Confidence"])
        df["Confidence"] = df["Confidence"].round(2)
        incident_placeholder.dataframe(df, use_container_width=True)
        alert_count_placeholder.metric("Total Alerts This Session", len(rows))
    else:
        incident_placeholder.info("No incidents detected yet.")

update_incidents_table()

if stop_button:
    st.session_state.running = False

if start_button:
    st.session_state.running = True

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error(f"Could not open video source: {video_path}")
            st.session_state.running = False
        else:
            active_alerts_dict = {}
            fps_display = st.sidebar.empty()
            frame_count = 0

            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.info("✅ End of video stream.")
                    break

                frame_count += 1

                # Skip frames for performance (especially for 4K/60fps video)
                if frame_count % frame_skip != 0:
                    continue

                t_start = time.time()

                # Resize large frames for faster processing
                h, w = frame.shape[:2]
                if w > process_width:
                    scale = process_width / w
                    frame = cv2.resize(frame, (process_width, int(h * scale)))

                # Process frame
                annotated_frame, new_alerts = tracker.process_frame(frame, active_alerts_dict)

                # Log any new alerts
                if new_alerts:
                    for alert in new_alerts:
                        log_incident(alert[0], alert[1], alert[2])
                    update_incidents_table()

                # Convert BGR to RGB for Streamlit
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                # Show FPS in sidebar
                elapsed = time.time() - t_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_display.metric("Processing FPS", f"{fps:.1f}")

            cap.release()
            st.session_state.running = False
    else:
        st.error("No video source selected. Please upload a video or provide a sample video.")
        st.session_state.running = False
