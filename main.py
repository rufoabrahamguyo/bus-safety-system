"""
Main Integration Script
Ties together all components: blind spot detection, drowsiness detection, BEV mapping, and emergency control
"""

import cv2
import numpy as np
import yaml
import time
import threading
from typing import Dict, Optional
import os

from blind_spot_detector import BlindSpotDetector
from drowsiness_detector import DrowsinessDetector
from bev_mapper import BEVMapper
from emergency_controller import EmergencyController


class BusSafetySystem:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the complete bus safety system.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        print("Initializing Bus Safety System...")
        
        # Blind spot detector
        self.blind_spot_detector = BlindSpotDetector(
            model_path=self.config['detection']['model'],
            confidence_threshold=self.config['detection']['confidence_threshold'],
            classes_of_interest=self.config['detection']['classes_of_interest']
        )
        print("✓ Blind spot detector initialized")
        
        # Drowsiness detector
        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=self.config['drowsiness']['ear_threshold'],
            consecutive_frames=self.config['drowsiness']['consecutive_frames'],
            head_tilt_threshold=self.config['drowsiness'].get('head_tilt_threshold', 25.0),
            head_tilt_consecutive=self.config['drowsiness'].get('head_tilt_consecutive', 5)
        )
        print("✓ Drowsiness detector initialized")
        
        # BEV mapper
        self.bev_mapper = BEVMapper(
            output_width=self.config['bev']['output_width'],
            output_height=self.config['bev']['output_height'],
            homography_dir=self.config['bev']['homography_dir']
        )
        print("✓ BEV mapper initialized")
        
        # Emergency controller
        self.emergency_controller = EmergencyController(
            enable_sound=self.config['emergency']['enable_sound_alert'],
            enable_visual=self.config['emergency']['enable_visual_alert'],
            enable_brake_mock=self.config['emergency']['enable_brake_mock'],
            arduino_port=self.config['emergency'].get('arduino_port'),
            brake_baudrate=self.config['emergency'].get('brake_baudrate', 9600)
        )
        print("✓ Emergency controller initialized")
        
        # Camera setup
        self.cameras = {}
        self._initialize_cameras()
        
        # State
        self.running = False
        self.latest_frames = {}
        self.latest_detections = {}
        self.latest_drowsiness = None
        self.latest_bev_map = None
        self._camera_warnings = set()  # Track camera warnings
        
        # Threading
        self.processing_thread = None
        
    def _initialize_cameras(self):
        """Initialize camera connections."""
        camera_config = self.config['cameras']
        
        for name, index in camera_config.items():
            try:
                # Support both integer indices (USB cameras) and URL strings (IP cameras/phones)
                if isinstance(index, str):
                    # It's a URL/IP address (phone camera)
                    cap = cv2.VideoCapture(index)
                    camera_info = f"URL: {index}"
                else:
                    # It's an integer camera index (USB camera)
                    cap = cv2.VideoCapture(index)
                    camera_info = f"index {index}"
                
                if cap.isOpened():
                    # Set resolution immediately for faster processing (before any delays)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS for consistency
                    
                    # For IP cameras, verify connection quickly (non-blocking check)
                    if isinstance(index, str):
                        # Quick test without blocking - camera will be verified during operation
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                        print(f"✓ Camera '{name}' connecting ({camera_info}) - will verify during operation")
                    else:
                        # USB cameras are usually instant
                        pass
                    
                    self.cameras[name] = cap
                    print(f"✓ Camera '{name}' initialized ({camera_info})")
                else:
                    print(f"✗ Could not open camera '{name}' ({camera_info})")
                    if isinstance(index, str):
                        print(f"   Check: 1) Phone is on same WiFi, 2) IP Webcam is running, 3) URL is correct")
                    self.cameras[name] = None
            except Exception as e:
                print(f"✗ Error initializing camera '{name}': {e}")
                self.cameras[name] = None
    
    def _read_camera_frames(self) -> Dict[str, np.ndarray]:
        """Read frames from all cameras."""
        frames = {}
        for name, cap in self.cameras.items():
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    frames[name] = frame
                    # Clear warning if camera recovers
                    if name in self._camera_warnings:
                        self._camera_warnings.remove(name)
                        print(f"✓ Camera '{name}' recovered and is working")
                else:
                    frames[name] = None
                    # Log if camera stops working (only once per camera)
                    if name not in self._camera_warnings:
                        camera_config = self.config['cameras'].get(name, 'unknown')
                        if isinstance(camera_config, str):
                            print(f"⚠️ Warning: Camera '{name}' (IP camera) stopped providing frames")
                            print(f"   Check if phone is still connected and IP Webcam is running")
                        else:
                            print(f"⚠️ Warning: Camera '{name}' (index {camera_config}) stopped providing frames")
                        self._camera_warnings.add(name)
            else:
                frames[name] = None
        return frames
    
    def _process_frame(self):
        """Main processing loop (runs in separate thread)."""
        while self.running:
            start_time = time.time()
            
            # Read camera frames
            frames = self._read_camera_frames()
            self.latest_frames = frames
            
            # Process blind spot detection
            detection_frames = {k: v for k, v in frames.items() if k != 'driver' and v is not None}
            if detection_frames:
                detections = self.blind_spot_detector.detect_multi_camera(detection_frames)
                self.latest_detections = detections
                
                # Check for dangerous objects
                danger_zone_radius = self.config['detection']['danger_zone_radius']
                for camera_name, detection_result in detections.items():
                    if detection_result:
                        dangerous = self.blind_spot_detector.get_dangerous_objects(
                            detection_result, danger_zone_radius
                        )
                        if dangerous:
                            print(f"⚠️ Dangerous objects detected from {camera_name}: {len(dangerous)}")
            
            # Process drowsiness detection
            driver_frame = frames.get('driver')
            if driver_frame is not None:
                drowsiness_result = self.drowsiness_detector.detect(driver_frame)
                self.latest_drowsiness = drowsiness_result
                
                # Debug output (print every 30 frames to avoid spam)
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                else:
                    self._frame_count = 0
                
                if self._frame_count % 30 == 0 and drowsiness_result['face_detected']:
                    print(f"Driver Status: EAR={drowsiness_result['ear_avg']:.3f}, "
                          f"Closed Frames={drowsiness_result['closed_eye_counter']}/{drowsiness_result['consecutive_frames_threshold']}, "
                          f"Drowsy={drowsiness_result['is_drowsy']}")
                
                # Trigger alert if drowsy
                if drowsiness_result['is_drowsy']:
                    print("⚠️ DROWSINESS DETECTED! Triggering alert...")
                    self.emergency_controller.trigger_drowsiness_alert()
                    
                    # Also check if there are dangerous objects nearby
                    if self.latest_detections:
                        has_dangerous = any(
                            self.blind_spot_detector.get_dangerous_objects(det, danger_zone_radius)
                            for det in self.latest_detections.values()
                            if det is not None
                        )
                        if has_dangerous:
                            self.emergency_controller.trigger_emergency_brake()
                
                # Trigger alert if head is tilted
                if drowsiness_result.get('head_tilted', False):
                    max_angle = max(abs(drowsiness_result.get('head_pitch', 0)),
                                  abs(drowsiness_result.get('head_yaw', 0)),
                                  abs(drowsiness_result.get('head_roll', 0)))
                    print(f"⚠️ HEAD TILT DETECTED! Angle: {max_angle:.1f}° - Triggering alert...")
                    self.emergency_controller.trigger_alert("head_tilt", "medium")
            else:
                # Driver camera not working
                if hasattr(self, '_driver_warning_count'):
                    self._driver_warning_count += 1
                else:
                    self._driver_warning_count = 0
                    print("⚠️ WARNING: Driver camera (index 4) not working! Drowsiness detection disabled.")
                    print("   Make sure camera index 4 is connected and has permission.")
            
            # Create BEV map
            detection_frames = {k: v for k, v in frames.items() if k != 'driver' and v is not None}
            if detection_frames:
                bev_map = self.bev_mapper.create_bev_map(detection_frames, self.latest_detections)
                self.latest_bev_map = bev_map
            
            # Control frame rate (adaptive based on processing time)
            elapsed = time.time() - start_time
            target_fps = 30
            sleep_time = max(0, (1.0 / target_fps) - elapsed)
            
            # Skip sleep if we're already behind (keeps system responsive)
            if elapsed < (1.0 / target_fps):
                time.sleep(sleep_time)
    
    def start(self):
        """Start the system."""
        if self.running:
            print("System is already running!")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frame, daemon=True)
        self.processing_thread.start()
        print("🚍 Bus Safety System started!")
    
    def stop(self):
        """Stop the system."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Release cameras
        for name, cap in self.cameras.items():
            if cap is not None:
                cap.release()
        
        # Cleanup emergency controller
        self.emergency_controller.cleanup()
        
        print("System stopped.")
    
    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            "running": self.running,
            "drowsiness": self.latest_drowsiness,
            "detections": {
                name: {
                    "count": len(det.get("boxes", [])) if det else 0,
                    "dangerous": len(self.blind_spot_detector.get_dangerous_objects(det)) if det else 0
                }
                for name, det in self.latest_detections.items()
            },
            "bev_map_available": self.latest_bev_map is not None
        }
    
    def display_preview(self):
        """Display preview windows (for testing without dashboard)."""
        if not self.running:
            print("System is not running. Call start() first.")
            return
        
        print("Displaying preview windows. Press 'q' to quit.")
        
        while self.running:
            # Display driver frame with drowsiness status
            if self.latest_drowsiness:
                driver_frame = self.latest_drowsiness.get('annotated_frame')
                if driver_frame is not None:
                    cv2.imshow('Driver Monitoring', driver_frame)
            
            # Display BEV map
            if self.latest_bev_map is not None:
                cv2.imshow('Bird\'s Eye View Map', self.latest_bev_map)
            
            # Display camera feeds with detections
            for camera_name, detection_result in self.latest_detections.items():
                if detection_result and camera_name in self.latest_frames:
                    frame = self.latest_frames[camera_name]
                    if frame is not None:
                        annotated = self.blind_spot_detector.draw_detections(frame, detection_result)
                        cv2.imshow(f'Camera: {camera_name}', annotated)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bus Safety System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview windows (for testing)')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found!")
        print("Please create a config.yaml file or specify a different config file.")
        return
    
    # Initialize system
    system = BusSafetySystem(config_path=args.config)
    
    try:
        # Start system
        system.start()
        
        if args.preview:
            # Display preview windows
            system.display_preview()
        else:
            # Run without preview (for use with dashboard)
            print("System running in background. Use dashboard.py to view.")
            print("Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop()


if __name__ == "__main__":
    main()

