import cv2
import time
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Cooldown in seconds before the same region triggers a new alert
ALERT_COOLDOWN = 5.0

# Minimum contour area (in pixels) to be considered a real object, not noise
MIN_CONTOUR_AREA = 3000

class RailwayTracker:
    """
    Railway intrusion detector using OpenCV background subtraction + contour analysis.
    
    This approach works by:
    1. Learning the static background of the scene (the empty track).
    2. Detecting any moving foreground objects (people, animals, debris).
    3. Checking if those moving objects enter the defined danger zone polygon.
    
    This requires NO model downloads and works entirely offline with OpenCV 5.0.
    """
    
    def __init__(self, model_path=None):
        # Background subtractor (MOG2) — learns the static scene over time
        # history=500: uses 500 frames to build the background model
        # varThreshold=50: sensitivity (lower = more sensitive, more noise)
        # detectShadows=True: helps separate shadows from real objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Morphological kernel for cleaning up the foreground mask
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Danger zone polygon (normalized coordinates, adjustable via UI)
        self.zone_polygon_normalized = [
            (0.3, 0.4), (0.7, 0.4),
            (1.0, 1.0), (0.0, 1.0)
        ]
        self.last_alert_time = 0.0
        self.object_counter = 0  # Simple counter for pseudo track IDs
        
    def get_pixel_polygon(self, frame_shape):
        """Convert normalized polygon coordinates to pixel coordinates."""
        h, w = frame_shape[:2]
        return [(int(x * w), int(y * h)) for x, y in self.zone_polygon_normalized]

    def is_point_in_zone(self, point, zone_geom):
        """Check if a point is inside a pre-computed Shapely Polygon."""
        return zone_geom.contains(Point(point))

    def classify_by_size(self, area):
        """Rough classification of detected object based on contour area."""
        if area > 15000:
            return "person/large_object"
        elif area > 8000:
            return "person/animal"
        elif area > 3000:
            return "small_object/animal"
        else:
            return "debris"

    def process_frame(self, frame, active_alerts_dict):
        """
        Process a single frame using background subtraction + contour analysis.
        
        Args:
            frame: BGR image from video capture
            active_alerts_dict: dict to track alerts (kept for API compatibility)
            
        Returns:
            annotated_frame: frame with drawings
            new_alerts: list of (class_name, track_id, confidence) tuples
        """
        pixel_polygon = self.get_pixel_polygon(frame.shape)
        zone_geom = Polygon(pixel_polygon)
        
        annotated_frame = frame.copy()
        new_alerts = []
        
        # --- Step 1: Apply background subtraction ---
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadow pixels are marked as 127 by MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Clean up noise with morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # --- Step 2: Find contours of moving objects ---
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- Step 3: Draw the danger zone polygon ---
        pts = np.array(pixel_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # Semi-transparent red overlay for the zone
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
        cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
        
        # Add zone label
        label_pos = pixel_polygon[0]
        cv2.putText(annotated_frame, "DANGER ZONE", (label_pos[0] + 5, label_pos[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        current_time = time.time()

        # --- Step 4: Analyze each detected moving object ---
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue  # Skip noise / tiny movements
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Bottom-center point (best proxy for "where the object stands")
            bottom_center = (int(x + w / 2), int(y + h))
            
            # Confidence is roughly proportional to contour area (normalized)
            confidence = min(area / 20000.0, 1.0)
            
            class_name = self.classify_by_size(area)
            
            cv2.circle(annotated_frame, bottom_center, 4, (255, 0, 0), -1)

            if self.is_point_in_zone(bottom_center, zone_geom):
                # INTRUSION DETECTED
                self.object_counter += 1
                track_id = self.object_counter
                
                # Log alert if cooldown has passed
                if current_time - self.last_alert_time > ALERT_COOLDOWN:
                    self.last_alert_time = current_time
                    new_alerts.append((class_name, track_id, round(confidence, 2)))
                    
                # RED bounding box + label for intrusion
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(annotated_frame, f"INTRUSION: {class_name}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Conf: {confidence:.0%}",
                            (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # GREEN bounding box for safe moving objects
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated_frame, class_name,
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- Step 5: Show detection stats on frame ---
        cv2.putText(annotated_frame, f"Moving Objects: {len([c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_frame, new_alerts
