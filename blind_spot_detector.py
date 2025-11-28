"""
360° Blind Spot Detection Module
Uses YOLOv8 for object detection and ByteTrack for tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict, Tuple, Optional
import time


class BlindSpotDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5,
                 classes_of_interest: List[int] = None):
        """
        Initialize the blind spot detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            classes_of_interest: List of COCO class IDs to detect (None = all)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.classes_of_interest = classes_of_interest
        
        # Initialize ByteTrack tracker
        self.tracker = sv.ByteTrack()
        
        # Detection history for each camera
        self.detection_history = {}
        
        # Class names mapping
        self.class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 7: "truck"
        }
        
    def detect(self, frame: np.ndarray, camera_name: str = "unknown") -> Dict:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input image frame (BGR format)
            camera_name: Name/ID of the camera (for tracking history)
            
        Returns:
            Dictionary containing detections, tracking IDs, and metadata
        """
        # Run YOLOv8 inference with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            classes=self.classes_of_interest,
            verbose=False
        )[0]
        
        # Convert to supervision Detections format
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter by confidence if needed
        detections = detections[detections.confidence >= self.confidence_threshold]
        
        # Store detection history
        if camera_name not in self.detection_history:
            self.detection_history[camera_name] = []
        
        # Prepare detection data
        detection_data = {
            "boxes": detections.xyxy,
            "track_ids": detections.tracker_id if detections.tracker_id is not None else [],
            "confidences": detections.confidence,
            "class_ids": detections.class_id,
            "class_names": [self.model.names[int(cls_id)] for cls_id in detections.class_id],
            "camera": camera_name,
            "timestamp": time.time()
        }
        
        return detection_data
    
    def detect_multi_camera(self, frames: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Detect objects in multiple camera frames simultaneously.
        
        Args:
            frames: Dictionary mapping camera names to frames
            
        Returns:
            Dictionary mapping camera names to their detection results
        """
        results = {}
        for camera_name, frame in frames.items():
            if frame is not None:
                results[camera_name] = self.detect(frame, camera_name)
            else:
                results[camera_name] = None
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: Dict, 
                      show_labels: bool = True, show_tracks: bool = True) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame: Input frame to draw on
            detections: Detection dictionary from detect() method
            show_labels: Whether to show class labels and confidence
            show_tracks: Whether to show tracking IDs
            
        Returns:
            Frame with detections drawn
        """
        annotated_frame = frame.copy()
        
        if detections is None or len(detections["boxes"]) == 0:
            return annotated_frame
        
        boxes = detections["boxes"]
        track_ids = detections["track_ids"]
        confidences = detections["confidences"]
        class_names = detections["class_names"]
        
        # Color mapping for different classes
        color_map = {
            "person": (0, 0, 255),      # Red
            "bicycle": (0, 255, 255),   # Yellow
            "motorcycle": (0, 255, 255), # Yellow
            "car": (255, 0, 0),         # Blue
            "bus": (255, 0, 255),       # Magenta
            "truck": (255, 0, 255),     # Magenta
        }
        
        for i, (box, conf, cls_name) in enumerate(zip(boxes, confidences, class_names)):
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this class
            color = color_map.get(cls_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if show_tracks and i < len(track_ids):
                label_parts.append(f"ID:{track_ids[i]}")
            if show_labels:
                label_parts.append(cls_name)
                label_parts.append(f"{conf:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = max(y1, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y - label_size[1] - 5),
                    (x1 + label_size[0], label_y + 5),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        return annotated_frame
    
    def get_dangerous_objects(self, detections: Dict, danger_zone_radius: float = 3.0) -> List[Dict]:
        """
        Identify objects that are dangerously close (within danger zone).
        
        Args:
            detections: Detection dictionary from detect() method
            danger_zone_radius: Radius in meters (approximate, based on pixel distance)
            
        Returns:
            List of dangerous object dictionaries
        """
        dangerous = []
        
        if detections is None or len(detections["boxes"]) == 0:
            return dangerous
        
        boxes = detections["boxes"]
        class_names = detections["class_names"]
        confidences = detections["confidences"]
        track_ids = detections["track_ids"]
        
        # Calculate approximate distance based on bounding box size
        # Larger boxes = closer objects (rough heuristic)
        for i, (box, cls_name, conf) in enumerate(zip(boxes, class_names, confidences)):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            
            # Heuristic: if box area > threshold, consider it dangerous
            # This is a simplified approach - in real system, use depth estimation
            area_threshold = 5000  # pixels^2 (adjust based on camera resolution)
            
            if box_area > area_threshold or cls_name in ["person", "bicycle", "motorcycle"]:
                dangerous.append({
                    "track_id": track_ids[i] if i < len(track_ids) else None,
                    "class": cls_name,
                    "confidence": float(conf),
                    "box": box.tolist(),
                    "camera": detections["camera"]
                })
        
        return dangerous

