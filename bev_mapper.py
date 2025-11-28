"""
Bird's Eye View (BEV) Mapping Module
Transforms multiple camera views into a top-down map using homography
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json


class BEVMapper:
    def __init__(self, output_width: int = 1000, output_height: int = 1000,
                 homography_dir: str = "homography_matrices"):
        """
        Initialize the BEV mapper.
        
        Args:
            output_width: Width of output BEV map in pixels
            output_height: Height of output BEV map in pixels
            homography_dir: Directory containing saved homography matrices
        """
        self.output_width = output_width
        self.output_height = output_height
        self.homography_dir = homography_dir
        
        # Create homography directory if it doesn't exist
        os.makedirs(homography_dir, exist_ok=True)
        
        # Load homography matrices for each camera
        self.homography_matrices = {}
        self.load_homographies()
        
        # BEV canvas
        self.bev_canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Camera positions in BEV space (for visualization)
        self.camera_positions = {
            "front": (output_width // 2, output_height - 50),
            "rear": (output_width // 2, 50),
            "left": (50, output_height // 2),
            "right": (output_width - 50, output_height // 2)
        }
        
    def load_homographies(self):
        """Load saved homography matrices from disk."""
        for camera_name in ["front", "rear", "left", "right"]:
            homography_path = os.path.join(self.homography_dir, f"H_{camera_name}.npy")
            if os.path.exists(homography_path):
                self.homography_matrices[camera_name] = np.load(homography_path)
                print(f"Loaded homography for {camera_name}")
            else:
                print(f"Warning: No homography matrix found for {camera_name}")
                # Create identity matrix as placeholder
                self.homography_matrices[camera_name] = np.eye(3)
    
    def save_homography(self, camera_name: str, homography: np.ndarray):
        """
        Save a homography matrix to disk.
        
        Args:
            camera_name: Name of the camera
            homography: 3x3 homography matrix
        """
        homography_path = os.path.join(self.homography_dir, f"H_{camera_name}.npy")
        np.save(homography_path, homography)
        self.homography_matrices[camera_name] = homography
        print(f"Saved homography for {camera_name}")
    
    def warp_to_bev(self, frame: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Warp a camera frame to bird's eye view.
        
        Args:
            frame: Input camera frame
            camera_name: Name of the camera (front, rear, left, right)
            
        Returns:
            Warped BEV frame
        """
        if camera_name not in self.homography_matrices:
            return np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        H = self.homography_matrices[camera_name]
        
        # Check if homography is valid (not identity)
        if np.allclose(H, np.eye(3)):
            return np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        warped = cv2.warpPerspective(
            frame,
            H,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return warped
    
    def transform_point_to_bev(self, point: Tuple[float, float], camera_name: str) -> Optional[Tuple[int, int]]:
        """
        Transform a point from camera coordinates to BEV coordinates.
        
        Args:
            point: (x, y) point in camera frame
            camera_name: Name of the camera
            
        Returns:
            (x, y) point in BEV coordinates, or None if transformation failed
        """
        if camera_name not in self.homography_matrices:
            return None
        
        H = self.homography_matrices[camera_name]
        
        # Check if homography is valid
        if np.allclose(H, np.eye(3)):
            return None
        
        # Transform point
        point_3d = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point_3d, H)
        
        if len(transformed) > 0:
            x, y = transformed[0][0]
            # Check if point is within BEV bounds
            if 0 <= x < self.output_width and 0 <= y < self.output_height:
                return (int(x), int(y))
        
        return None
    
    def create_bev_map(self, frames: Dict[str, np.ndarray], 
                      detections: Dict[str, Dict] = None) -> np.ndarray:
        """
        Create a complete BEV map from multiple camera frames and detections.
        
        Args:
            frames: Dictionary mapping camera names to frames
            detections: Optional dictionary mapping camera names to detection results
            
        Returns:
            Complete BEV map with detections overlaid
        """
        # Reset canvas
        self.bev_canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # Draw grid for reference
        self._draw_grid()
        
        # Draw bus outline in center
        self._draw_bus_outline()
        
        # Warp and blend camera views
        for camera_name in ["front", "rear", "left", "right"]:
            if camera_name in frames and frames[camera_name] is not None:
                warped = self.warp_to_bev(frames[camera_name], camera_name)
                
                # Blend warped view onto canvas (simple alpha blending)
                mask = warped.sum(axis=2) > 0
                self.bev_canvas[mask] = (self.bev_canvas[mask] * 0.3 + warped[mask] * 0.7).astype(np.uint8)
        
        # Draw detections on BEV map
        if detections:
            self._draw_detections_on_bev(detections)
        
        # Draw camera positions
        self._draw_camera_positions()
        
        return self.bev_canvas
    
    def _draw_grid(self):
        """Draw a grid on the BEV canvas for reference."""
        grid_spacing = 100
        color = (50, 50, 50)
        
        # Vertical lines
        for x in range(0, self.output_width, grid_spacing):
            cv2.line(self.bev_canvas, (x, 0), (x, self.output_height), color, 1)
        
        # Horizontal lines
        for y in range(0, self.output_height, grid_spacing):
            cv2.line(self.bev_canvas, (0, y), (self.output_width, y), color, 1)
    
    def _draw_bus_outline(self):
        """Draw a bus outline in the center of the BEV map."""
        center_x = self.output_width // 2
        center_y = self.output_height // 2
        bus_width = 200
        bus_length = 300
        
        # Draw bus rectangle
        pt1 = (center_x - bus_width // 2, center_y - bus_length // 2)
        pt2 = (center_x + bus_width // 2, center_y + bus_length // 2)
        cv2.rectangle(self.bev_canvas, pt1, pt2, (100, 100, 100), -1)
        cv2.rectangle(self.bev_canvas, pt1, pt2, (200, 200, 200), 2)
        
        # Draw front indicator
        cv2.circle(self.bev_canvas, (center_x, center_y - bus_length // 2), 10, (0, 255, 0), -1)
    
    def _draw_detections_on_bev(self, detections: Dict[str, Dict]):
        """
        Draw detection boxes on the BEV map.
        
        Args:
            detections: Dictionary mapping camera names to detection results
        """
        color_map = {
            "person": (0, 0, 255),      # Red
            "bicycle": (0, 255, 255),   # Yellow
            "motorcycle": (0, 255, 255), # Yellow
            "car": (255, 0, 0),         # Blue
            "bus": (255, 0, 255),       # Magenta
            "truck": (255, 0, 255),     # Magenta
        }
        
        for camera_name, detection_result in detections.items():
            if detection_result is None or len(detection_result.get("boxes", [])) == 0:
                continue
            
            boxes = detection_result["boxes"]
            class_names = detection_result["class_names"]
            confidences = detection_result["confidences"]
            track_ids = detection_result.get("track_ids", [])
            
            for i, (box, cls_name, conf) in enumerate(zip(boxes, class_names, confidences)):
                x1, y1, x2, y2 = box
                
                # Transform box center to BEV
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                bev_point = self.transform_point_to_bev((center_x, center_y), camera_name)
                
                if bev_point:
                    # Get color
                    color = color_map.get(cls_name, (0, 255, 0))
                    
                    # Draw circle at detection location
                    radius = 15
                    cv2.circle(self.bev_canvas, bev_point, radius, color, -1)
                    cv2.circle(self.bev_canvas, bev_point, radius, (255, 255, 255), 2)
                    
                    # Draw label
                    label = f"{cls_name[:3]}"
                    if i < len(track_ids):
                        label += f"#{track_ids[i]}"
                    
                    cv2.putText(
                        self.bev_canvas,
                        label,
                        (bev_point[0] + 20, bev_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
    
    def _draw_camera_positions(self):
        """Draw camera positions on the BEV map."""
        for camera_name, pos in self.camera_positions.items():
            cv2.circle(self.bev_canvas, pos, 15, (255, 255, 0), -1)
            cv2.putText(
                self.bev_canvas,
                camera_name.upper(),
                (pos[0] - 30, pos[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

