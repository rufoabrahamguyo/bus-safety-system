"""
Camera Calibration Tool for Bird's Eye View Homography
Helps calibrate each camera's homography matrix using a chessboard pattern
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional
import argparse


class HomographyCalibrator:
    def __init__(self, camera_index: int, camera_name: str, 
                 chessboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 1.0):
        """
        Initialize the homography calibrator.
        
        Args:
            camera_index: Camera device index
            camera_name: Name of the camera (front, rear, left, right)
            chessboard_size: Number of inner corners (width, height)
            square_size: Size of chessboard squares in meters (for reference)
        """
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Calibration data
        self.src_points = []  # Points in camera image
        self.dst_points = []  # Points in BEV space
        
        # Homography matrix
        self.homography = None
        
    def capture_chessboard(self) -> Optional[np.ndarray]:
        """
        Capture and detect chessboard pattern.
        
        Returns:
            Frame with chessboard corners drawn, or None if not detected
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            frame_with_corners = frame.copy()
            cv2.drawChessboardCorners(frame_with_corners, self.chessboard_size, corners, ret)
            
            return frame_with_corners, corners
        else:
            return frame, None
    
    def set_source_points(self, corners: np.ndarray):
        """
        Set source points from chessboard corners.
        Uses the four outer corners of the chessboard.
        
        Args:
            corners: Detected chessboard corners
        """
        # Reshape corners
        corners = corners.reshape(-1, 2)
        
        # Get four corners (assuming chessboard is rectangular)
        # Top-left, top-right, bottom-right, bottom-left
        h, w = self.chessboard_size
        
        # Map indices to corners
        top_left_idx = 0
        top_right_idx = w - 1
        bottom_left_idx = (h - 1) * w
        bottom_right_idx = h * w - 1
        
        self.src_points = np.array([
            corners[top_left_idx],
            corners[top_right_idx],
            corners[bottom_right_idx],
            corners[bottom_left_idx]
        ], dtype=np.float32)
        
        print(f"Source points set for {self.camera_name}:")
        for i, pt in enumerate(self.src_points):
            print(f"  Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    def set_destination_points(self, bev_width: int = 1000, bev_height: int = 1000):
        """
        Set destination points in BEV space.
        User can adjust these based on camera position.
        
        Args:
            bev_width: Width of BEV output
            bev_height: Height of BEV output
        """
        print(f"\nSetting destination points for {self.camera_name} camera.")
        print("These define where the camera view maps to in the BEV space.")
        print("Default positions:")
        
        # Default positions based on camera name
        defaults = {
            "front": [
                (bev_width // 2 - 200, bev_height - 100),  # Top-left
                (bev_width // 2 + 200, bev_height - 100),  # Top-right
                (bev_width // 2 + 200, bev_height - 400),  # Bottom-right
                (bev_width // 2 - 200, bev_height - 400),  # Bottom-left
            ],
            "rear": [
                (bev_width // 2 - 200, 100),   # Top-left
                (bev_width // 2 + 200, 100),    # Top-right
                (bev_width // 2 + 200, 400),    # Bottom-right
                (bev_width // 2 - 200, 400),    # Bottom-left
            ],
            "left": [
                (100, bev_height // 2 - 200),   # Top-left
                (400, bev_height // 2 - 200),   # Top-right
                (400, bev_height // 2 + 200),   # Bottom-right
                (100, bev_height // 2 + 200),   # Bottom-left
            ],
            "right": [
                (bev_width - 400, bev_height // 2 - 200),  # Top-left
                (bev_width - 100, bev_height // 2 - 200),  # Top-right
                (bev_width - 100, bev_height // 2 + 200),  # Bottom-right
                (bev_width - 400, bev_height // 2 + 200),  # Bottom-left
            ]
        }
        
        if self.camera_name in defaults:
            self.dst_points = np.array(defaults[self.camera_name], dtype=np.float32)
            print(f"Using default positions for {self.camera_name} camera.")
        else:
            # Manual input
            print("Enter destination points (x, y) for each corner:")
            self.dst_points = []
            for i in ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]:
                x = int(input(f"{i} X: ") or bev_width // 2)
                y = int(input(f"{i} Y: ") or bev_height // 2)
                self.dst_points.append([x, y])
            self.dst_points = np.array(self.dst_points, dtype=np.float32)
        
        print(f"Destination points set:")
        for i, pt in enumerate(self.dst_points):
            print(f"  Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    def calculate_homography(self) -> Optional[np.ndarray]:
        """
        Calculate homography matrix from source to destination points.
        
        Returns:
            3x3 homography matrix, or None if calculation fails
        """
        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            print("Error: Need exactly 4 source and 4 destination points")
            return None
        
        self.homography = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        return self.homography
    
    def test_homography(self, frame: np.ndarray) -> np.ndarray:
        """
        Test homography by warping a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Warped frame
        """
        if self.homography is None:
            return frame
        
        h, w = 1000, 1000  # BEV dimensions
        warped = cv2.warpPerspective(frame, self.homography, (w, h))
        return warped
    
    def save_homography(self, output_dir: str = "homography_matrices"):
        """
        Save homography matrix to disk.
        
        Args:
            output_dir: Directory to save homography matrix
        """
        if self.homography is None:
            print("Error: No homography matrix calculated")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"H_{self.camera_name}.npy")
        np.save(output_path, self.homography)
        print(f"✓ Homography saved to {output_path}")
    
    def cleanup(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()


def interactive_calibration(camera_index: int, camera_name: str):
    """
    Interactive calibration process.
    
    Args:
        camera_index: Camera device index
        camera_name: Name of the camera
    """
    calibrator = HomographyCalibrator(camera_index, camera_name)
    
    print(f"\n{'='*60}")
    print(f"Calibrating {camera_name.upper()} camera")
    print(f"{'='*60}\n")
    print("Instructions:")
    print("1. Place a chessboard pattern (9x6 inner corners) in the camera view")
    print("2. Press SPACE to capture when chessboard is detected")
    print("3. Press 'q' to quit")
    print("\nStarting camera feed...\n")
    
    try:
        while True:
            result = calibrator.capture_chessboard()
            
            if result is None:
                continue
            
            frame, corners = result
            
            if corners is not None:
                cv2.putText(
                    frame,
                    "Chessboard detected! Press SPACE to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No chessboard detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow(f"Calibration - {camera_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar
                if corners is not None:
                    calibrator.set_source_points(corners)
                    calibrator.set_destination_points()
                    calibrator.calculate_homography()
                    
                    # Test homography
                    test_warped = calibrator.test_homography(frame)
                    cv2.imshow(f"Warped - {camera_name}", test_warped)
                    
                    # Ask to save
                    save = input("\nSave this homography? (y/n): ").lower()
                    if save == 'y':
                        calibrator.save_homography()
                        print("Calibration complete!\n")
                        break
                    else:
                        print("Calibration discarded. Try again.\n")
                else:
                    print("No chessboard detected. Cannot capture.")
            
            elif key == ord('q'):
                print("Calibration cancelled.")
                break
        
    finally:
        calibrator.cleanup()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Calibrate camera homography for BEV mapping')
    parser.add_argument('--camera', type=int, required=True,
                       help='Camera index (0, 1, 2, etc.)')
    parser.add_argument('--name', type=str, required=True,
                       choices=['front', 'rear', 'left', 'right'],
                       help='Camera name')
    
    args = parser.parse_args()
    
    interactive_calibration(args.camera, args.name)


if __name__ == "__main__":
    main()

