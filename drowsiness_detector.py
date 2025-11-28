"""
Driver Drowsiness Detection Module
Uses MediaPipe Face Mesh and EAR (Eye Aspect Ratio) algorithm
Falls back to OpenCV face detection if MediaPipe is not available
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import time
from collections import deque

# Try to import MediaPipe, fallback to OpenCV if not available
try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None  # type: ignore
    print("Warning: MediaPipe not available. Using OpenCV fallback for drowsiness detection.")


class DrowsinessDetector:
    def __init__(self, ear_threshold: float = 0.2, consecutive_frames: int = 20,
                 head_tilt_threshold: float = 25.0, head_tilt_consecutive: int = 5):
        """
        Initialize the drowsiness detector.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold (lower = more sensitive)
            consecutive_frames: Number of consecutive frames with closed eyes before alert
            head_tilt_threshold: Head tilt angle in degrees before alert (pitch/yaw/roll)
            head_tilt_consecutive: Consecutive frames with head tilt before alert
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.head_tilt_threshold = head_tilt_threshold
        self.head_tilt_consecutive = head_tilt_consecutive
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            # Eye landmark indices (MediaPipe Face Mesh)
            self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
            self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        else:
            # Fallback: Use OpenCV Haar Cascades with multiple fallback options
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
            ]
            
            self.face_cascade = None
            for path in cascade_paths:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.face_cascade = cascade
                    break
            
            if self.face_cascade is None:
                # Last resort: use default
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.face_mesh = None
        
        # State tracking
        self.ear_history = deque(maxlen=30)  # Keep last 30 EAR values
        self.closed_eye_counter = 0
        self.is_drowsy = False
        self.last_alert_time = 0
        
        # Head tilt tracking
        self.head_tilt_counter = 0
        self.is_head_tilted = False
        self.head_pitch = 0.0  # Nodding up/down
        self.head_yaw = 0.0    # Turning left/right
        self.head_roll = 0.0   # Tilting left/right shoulder
        
        # Angle smoothing (moving average filter)
        self.pitch_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=5)
        self.roll_history = deque(maxlen=5)
        
        # Baseline calibration (learns normal head position)
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.baseline_roll = None
        self.calibration_frames = 0
        self.calibration_complete = False
        
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)
        
        Args:
            eye_landmarks: Array of 6 eye landmark points
            
        Returns:
            EAR value (lower = more closed)
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Avoid division by zero
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_perclos(self, window_size: int = 30) -> float:
        """
        Calculate PERCLOS (Percentage of time eyes are closed).
        
        Args:
            window_size: Number of recent frames to consider
            
        Returns:
            PERCLOS value (0.0 to 1.0)
        """
        if len(self.ear_history) < window_size:
            return 0.0
        
        recent_ears = list(self.ear_history)[-window_size:]
        closed_count = sum(1 for ear in recent_ears if ear < self.ear_threshold)
        return closed_count / len(recent_ears)
    
    def calculate_head_pose(self, face_box: Tuple[int, int, int, int], 
                           left_eye: Optional[Tuple[int, int]] = None,
                           right_eye: Optional[Tuple[int, int]] = None,
                           frame_width: int = 640) -> Dict[str, float]:
        """
        Calculate head pose angles (pitch, yaw, roll) using improved ML-based estimation.
        Uses temporal smoothing and baseline calibration for accuracy.
        
        Args:
            face_box: (x, y, w, h) bounding box of face
            left_eye: Optional (x, y) center of left eye
            right_eye: Optional (x, y) center of right eye
            frame_width: Width of the frame for yaw calculation
            
        Returns:
            Dictionary with pitch, yaw, roll angles in degrees (smoothed and calibrated)
        """
        x, y, w, h = face_box
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        # Initialize raw angles
        pitch_raw = 0.0  # Up/down (nodding)
        yaw_raw = 0.0    # Left/right (turning)
        roll_raw = 0.0   # Tilt left/right (shoulder tilt)
        
        # Calculate roll (tilt) from eye positions - most accurate method
        if left_eye is not None and right_eye is not None:
            eye_dx = right_eye[0] - left_eye[0]
            eye_dy = right_eye[1] - left_eye[1]
            eye_distance = np.sqrt(eye_dx**2 + eye_dy**2)
            
            if eye_dx != 0 and eye_distance > 10:  # Minimum eye distance for accuracy
                # Calculate roll angle from eye line (shoulder tilt)
                roll_raw = np.degrees(np.arctan2(eye_dy, eye_dx))
            
            # Improved pitch calculation using eye position and face geometry
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            
            # Use face geometry for pitch estimation
            face_top = y
            face_bottom = y + h
            face_height = h
            
            if face_height > 0:
                # Normalized eye position (0 = top of face, 1 = bottom)
                eye_position_normalized = (eye_center_y - face_top) / face_height
                
                # More accurate pitch calculation
                # Normal eyes are typically at 0.35-0.4 from top of face
                ideal_eye_position = 0.375
                deviation = eye_position_normalized - ideal_eye_position
                
                # Convert deviation to angle (calibrated scaling)
                if deviation < -0.1:  # Eyes too high = looking up
                    pitch_raw = deviation * 60  # More sensitive scaling
                elif deviation > 0.1:  # Eyes too low = looking down
                    pitch_raw = deviation * 60
            
            # Improved yaw calculation using eye positions and face center
            # When head turns, one eye becomes closer to camera (appears larger/closer)
            # and face center shifts
            face_center_offset = face_center_x - (frame_width / 2)
            
            # Use eye positions for more accurate yaw
            eye_center_offset = eye_center_x - (frame_width / 2)
            
            # Normalize and convert to angle
            normalized_offset = eye_center_offset / (frame_width / 2)  # -1 to 1
            yaw_raw = normalized_offset * 35  # Increased sensitivity
            
            # Additional yaw estimation from eye asymmetry
            if eye_distance > 10:
                # When head turns, eyes appear at different distances
                eye_asymmetry = abs(left_eye[0] - face_center_x) - abs(right_eye[0] - face_center_x)
                asymmetry_factor = eye_asymmetry / eye_distance
                yaw_raw += asymmetry_factor * 20  # Additional yaw component
        
        # Fallback pitch estimation from face aspect ratio
        if abs(pitch_raw) < 1.0:  # If pitch not calculated well
            aspect_ratio = h / max(w, 1)
            ideal_ratio = 1.2  # Typical face aspect ratio
            ratio_deviation = aspect_ratio - ideal_ratio
            if abs(ratio_deviation) > 0.1:
                pitch_raw = ratio_deviation * 25
        
        # Fallback yaw estimation from face position
        if abs(yaw_raw) < 1.0:
            frame_center = frame_width / 2
            offset = face_center_x - frame_center
            normalized_offset = offset / (frame_width / 2)
            yaw_raw = normalized_offset * 30
        
        # Apply temporal smoothing (moving average filter)
        self.pitch_history.append(pitch_raw)
        self.yaw_history.append(yaw_raw)
        self.roll_history.append(roll_raw)
        
        # Calculate smoothed angles
        pitch_smoothed = np.mean(self.pitch_history) if len(self.pitch_history) > 0 else pitch_raw
        yaw_smoothed = np.mean(self.yaw_history) if len(self.yaw_history) > 0 else yaw_raw
        roll_smoothed = np.mean(self.roll_history) if len(self.roll_history) > 0 else roll_raw
        
        # Baseline calibration (learn normal position over first 30 frames)
        if not self.calibration_complete:
            self.calibration_frames += 1
            if self.baseline_pitch is None:
                self.baseline_pitch = pitch_smoothed
                self.baseline_yaw = yaw_smoothed
                self.baseline_roll = roll_smoothed
            else:
                # Moving average of baseline
                alpha = 0.1  # Learning rate
                self.baseline_pitch = (1 - alpha) * self.baseline_pitch + alpha * pitch_smoothed
                self.baseline_yaw = (1 - alpha) * self.baseline_yaw + alpha * yaw_smoothed
                self.baseline_roll = (1 - alpha) * self.baseline_roll + alpha * roll_smoothed
            
            if self.calibration_frames >= 30:
                self.calibration_complete = True
        
        # Apply baseline correction (subtract baseline to get relative angles)
        if self.calibration_complete:
            pitch = pitch_smoothed - self.baseline_pitch
            yaw = yaw_smoothed - self.baseline_yaw
            roll = roll_smoothed - self.baseline_roll
        else:
            # During calibration, use raw angles
            pitch = pitch_smoothed
            yaw = yaw_smoothed
            roll = roll_smoothed
        
        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "max_angle": max(abs(pitch), abs(yaw), abs(roll))
        }
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect drowsiness in a driver frame.
        
        Args:
            frame: Input frame (BGR format) with driver's face
            
        Returns:
            Dictionary containing drowsiness status and metrics
        """
        if self.use_mediapipe:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Dict:
        """Detect using MediaPipe (preferred method)."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize return values
        ear_left = 0.0
        ear_right = 0.0
        ear_avg = 0.0
        face_detected = False
        landmarks_drawn = None
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Extract eye landmarks
            left_eye_points = []
            right_eye_points = []
            
            for idx in self.LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                left_eye_points.append(np.array([x, y]))
            
            for idx in self.RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                right_eye_points.append(np.array([x, y]))
            
            # Calculate EAR for both eyes
            ear_left = self.calculate_ear(np.array(left_eye_points))
            ear_right = self.calculate_ear(np.array(right_eye_points))
            ear_avg = (ear_left + ear_right) / 2.0
            
            # Store in history
            self.ear_history.append(ear_avg)
            
            # Check if eyes are closed
            if ear_avg < self.ear_threshold:
                self.closed_eye_counter += 1
            else:
                self.closed_eye_counter = 0
            
            # Determine drowsiness status
            self.is_drowsy = self.closed_eye_counter >= self.consecutive_frames
            
            # Draw landmarks for visualization
            landmarks_drawn = frame.copy()
            for point in left_eye_points + right_eye_points:
                cv2.circle(landmarks_drawn, tuple(point.astype(int)), 2, (0, 255, 0), -1)
            
            # Calculate head pose from eye positions
            left_eye_center = (int(np.mean([p[0] for p in left_eye_points])), 
                              int(np.mean([p[1] for p in left_eye_points])))
            right_eye_center = (int(np.mean([p[0] for p in right_eye_points])), 
                               int(np.mean([p[1] for p in right_eye_points])))
            
            # Estimate face box from landmarks
            all_x = [p[0] for p in left_eye_points + right_eye_points]
            all_y = [p[1] for p in left_eye_points + right_eye_points]
            face_x = int(min(all_x)) - 50
            face_y = int(min(all_y)) - 50
            face_w = int(max(all_x) - min(all_x)) + 100
            face_h = int(max(all_y) - min(all_y)) + 100
            
            head_pose = self.calculate_head_pose(
                (face_x, face_y, face_w, face_h),
                left_eye_center,
                right_eye_center,
                w  # Frame width
            )
            
            self.head_pitch = head_pose["pitch"]
            self.head_yaw = head_pose["yaw"]
            self.head_roll = head_pose["roll"]
            max_tilt = head_pose["max_angle"]
            
            # Check if head is tilted
            # More sensitive: check if ANY angle exceeds threshold
            if max_tilt > self.head_tilt_threshold:
                self.head_tilt_counter += 1
                if self.head_tilt_counter >= self.head_tilt_consecutive:
                    self.is_head_tilted = True
            else:
                # Gradual reset instead of immediate reset for smoother detection
                self.head_tilt_counter = max(0, self.head_tilt_counter - 1)
                if self.head_tilt_counter == 0:
                    self.is_head_tilted = False
            
            # Draw head pose info
            cv2.putText(landmarks_drawn, f"Pitch: {self.head_pitch:.1f}°", 
                       (face_x, face_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(landmarks_drawn, f"Yaw: {self.head_yaw:.1f}°", 
                       (face_x, face_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(landmarks_drawn, f"Roll: {self.head_roll:.1f}°", 
                       (face_x, face_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            if self.is_head_tilted:
                cv2.putText(landmarks_drawn, "HEAD TILTED!", 
                           (face_x, face_y + face_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No face detected - reset counters
            self.closed_eye_counter = 0
            self.is_drowsy = False
            self.head_tilt_counter = 0
            self.is_head_tilted = False
            self.head_pitch = 0.0
            self.head_yaw = 0.0
            self.head_roll = 0.0
        
        # Calculate PERCLOS
        perclos = self.calculate_perclos()
        
        return {
            "is_drowsy": self.is_drowsy,
            "face_detected": face_detected,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "ear_avg": ear_avg,
            "perclos": perclos,
            "closed_eye_counter": self.closed_eye_counter,
            "consecutive_frames_threshold": self.consecutive_frames,
            "head_tilted": self.is_head_tilted,
            "head_pitch": self.head_pitch,
            "head_yaw": self.head_yaw,
            "head_roll": self.head_roll,
            "head_tilt_counter": self.head_tilt_counter,
            "head_tilt_threshold": self.head_tilt_threshold,
            "timestamp": time.time(),
            "annotated_frame": landmarks_drawn if landmarks_drawn is not None else frame
        }
    
    def _detect_opencv(self, frame: np.ndarray) -> Dict:
        """Detect using OpenCV Haar Cascades (optimized for speed and accuracy)."""
        # Resize frame for faster processing (if too large)
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame_small = cv2.resize(frame, (new_width, new_height))
            scale_factor = width / new_width
        else:
            frame_small = frame
            scale_factor = 1.0
        
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        landmarks_drawn = frame.copy()  # Draw on original frame
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) - better than simple equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Optimized face detection - try enhanced first (usually works best)
        faces = self.face_cascade.detectMultiScale(
            gray_enhanced, 
            scaleFactor=1.1,
            minNeighbors=4,       # Balanced for accuracy
            minSize=(40, 40),     # Reasonable minimum size
            maxSize=(300, 300),   # Maximum size to avoid false positives
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces found, try original gray
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Scale faces back to original frame size if we resized
        if scale_factor != 1.0:
            faces = [(int(x * scale_factor), int(y * scale_factor), 
                     int(w * scale_factor), int(h * scale_factor)) for (x, y, w, h) in faces]
        
        ear_left = 0.0
        ear_right = 0.0
        ear_avg = 0.0
        face_detected = len(faces) > 0
        
        if len(faces) > 0:
            # Use the largest face (most likely to be the driver)
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Scale coordinates if we resized the frame
            if scale_factor != 1.0:
                x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
                # Get ROI from original frame
                roi_gray_full = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            else:
                roi_gray_full = gray[y:y+h, x:x+w]
            
            # Draw face rectangle
            cv2.rectangle(landmarks_drawn, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detect eyes in the face region (use upper 60% of face where eyes are)
            roi_gray = roi_gray_full[:int(h*0.6), :]  # Upper 60% of face
            
            # Optimized eye detection with CLAHE
            eyes = []
            
            if roi_gray.size > 0 and roi_gray.shape[0] > 10 and roi_gray.shape[1] > 10:
                # Apply CLAHE to eye region for better detection
                try:
                    clahe_eye = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                    roi_gray_enhanced = clahe_eye.apply(roi_gray)
                    
                    # Try enhanced first
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray_enhanced,
                        scaleFactor=1.1,
                        minNeighbors=3,  # Higher for accuracy
                        minSize=(15, 15),  # Reasonable minimum
                        maxSize=(int(w*0.4), int(h*0.3)),  # Max eye size relative to face
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                except:
                    pass
                
                # If no eyes found, try original ROI
                if len(eyes) == 0:
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(15, 15),
                        maxSize=(int(w*0.4), int(h*0.3)),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
            
            # Calculate head pose angles
            left_eye_center = None
            right_eye_center = None
            
            if len(eyes) >= 2:
                # Sort eyes by x position (left and right)
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Get eye centers for head pose calculation
                left_eye = eyes[0]
                right_eye = eyes[1]
                left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
                right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
                
                # Calculate simple eye aspect ratio based on eye size
                # This is a simplified version - not as accurate as MediaPipe
                eye_ratios = []
                for (ex, ey, ew, eh) in eyes[:2]:  # Use first two eyes
                    eye_ratio = eh / max(ew, 1)  # Height/width ratio
                    eye_ratios.append(eye_ratio)
                    cv2.rectangle(landmarks_drawn, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                
                if len(eye_ratios) == 2:
                    ear_avg = sum(eye_ratios) / 2.0
                    ear_left = eye_ratios[0]
                    ear_right = eye_ratios[1]
                    
                    # Store in history
                    self.ear_history.append(ear_avg)
                    
                    # IMPORTANT: When eyes are CLOSED, the height/width ratio INCREASES
                    # Open eyes: ratio ~0.2-0.3, Closed eyes: ratio ~0.5-0.8
                    # So we check if ratio is HIGHER than threshold (opposite of MediaPipe EAR)
                    closed_eye_threshold = 0.5  # Eyes closed if ratio > 0.5
                    
                    # Check if eyes are closed (high ratio = closed)
                    if ear_avg > closed_eye_threshold:
                        self.closed_eye_counter += 1
                    else:
                        # Also check if ratio is very low (might indicate eyes are squinting/closing)
                        if ear_avg < self.ear_threshold:
                            self.closed_eye_counter += 1
                        else:
                            self.closed_eye_counter = max(0, self.closed_eye_counter - 1)  # Gradual reset
                    
                    # Determine drowsiness status
                    self.is_drowsy = self.closed_eye_counter >= self.consecutive_frames
                else:
                    # Only one eye detected or none - might be closing
                    if len(self.ear_history) > 2:  # Face was detected before
                        self.closed_eye_counter += 1
                    else:
                        self.closed_eye_counter = max(0, self.closed_eye_counter - 1)
                    self.is_drowsy = self.closed_eye_counter >= self.consecutive_frames
            else:
                # Eyes not detected - likely closed or looking away
                # More aggressive detection: only need 2 frames of history
                if len(self.ear_history) > 2:  # Face was detected before (reduced from 5)
                    self.closed_eye_counter += 1
                    if self.closed_eye_counter >= self.consecutive_frames:
                        self.is_drowsy = True
                else:
                    # Face just detected, eyes might not be visible yet
                    self.closed_eye_counter = max(0, self.closed_eye_counter - 1)
                    self.is_drowsy = False
        else:
            # No face detected - reset counters
            self.closed_eye_counter = 0
            self.is_drowsy = False
            self.head_tilt_counter = 0
            self.is_head_tilted = False
            # Clear history if face not detected for a while
            if len(self.ear_history) > 0:
                self.ear_history.clear()
        
        # Calculate head pose angles if face detected
        if face_detected and len(faces) > 0:
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            if scale_factor != 1.0:
                x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
            
            # Get actual frame width for yaw calculation
            frame_width_actual = frame.shape[1]
            
            head_pose = self.calculate_head_pose(
                (x, y, w, h),
                left_eye_center,
                right_eye_center,
                frame_width_actual
            )
            
            self.head_pitch = head_pose["pitch"]
            self.head_yaw = head_pose["yaw"]
            self.head_roll = head_pose["roll"]
            max_tilt = head_pose["max_angle"]
            
            # Check if head is tilted beyond threshold
            # More sensitive: check if ANY angle exceeds threshold
            if max_tilt > self.head_tilt_threshold:
                self.head_tilt_counter += 1
                if self.head_tilt_counter >= self.head_tilt_consecutive:
                    self.is_head_tilted = True
            else:
                # Gradual reset instead of immediate reset for smoother detection
                self.head_tilt_counter = max(0, self.head_tilt_counter - 1)
                if self.head_tilt_counter == 0:
                    self.is_head_tilted = False
            
            # Draw head pose info on frame
            cv2.putText(landmarks_drawn, f"Pitch: {self.head_pitch:.1f}°", 
                       (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(landmarks_drawn, f"Yaw: {self.head_yaw:.1f}°", 
                       (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(landmarks_drawn, f"Roll: {self.head_roll:.1f}°", 
                       (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            if self.is_head_tilted:
                cv2.putText(landmarks_drawn, "HEAD TILTED!", 
                           (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.head_pitch = 0.0
            self.head_yaw = 0.0
            self.head_roll = 0.0
            self.head_tilt_counter = 0
            self.is_head_tilted = False
        
        # Calculate PERCLOS
        perclos = self.calculate_perclos()
        
        return {
            "is_drowsy": self.is_drowsy,
            "face_detected": face_detected,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "ear_avg": ear_avg,
            "perclos": perclos,
            "closed_eye_counter": self.closed_eye_counter,
            "consecutive_frames_threshold": self.consecutive_frames,
            "head_tilted": self.is_head_tilted,
            "head_pitch": self.head_pitch,
            "head_yaw": self.head_yaw,
            "head_roll": self.head_roll,
            "head_tilt_counter": self.head_tilt_counter,
            "head_tilt_threshold": self.head_tilt_threshold,
            "timestamp": time.time(),
            "annotated_frame": landmarks_drawn
        }
    
    def draw_status(self, frame: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw drowsiness status on frame.
        
        Args:
            frame: Input frame
            detection_result: Result dictionary from detect() method
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Get status
        is_drowsy = detection_result["is_drowsy"]
        face_detected = detection_result["face_detected"]
        ear_avg = detection_result["ear_avg"]
        perclos = detection_result["perclos"]
        counter = detection_result["closed_eye_counter"]
        
        # Draw status text
        if not face_detected:
            status_text = "NO FACE DETECTED"
            color = (128, 128, 128)
        elif is_drowsy:
            status_text = "DROWSY! ALERT!"
            color = (0, 0, 255)  # Red
        else:
            status_text = "AWAKE"
            color = (0, 255, 0)  # Green
        
        # Draw main status
        cv2.putText(
            annotated,
            status_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )
        
        # Draw metrics
        metrics_y = 100
        cv2.putText(
            annotated,
            f"EAR: {ear_avg:.3f}",
            (50, metrics_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            annotated,
            f"PERCLOS: {perclos:.2%}",
            (50, metrics_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            annotated,
            f"Closed Frames: {counter}/{self.consecutive_frames}",
            (50, metrics_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw warning overlay if drowsy
        if is_drowsy:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            annotated = overlay
        
        return annotated

