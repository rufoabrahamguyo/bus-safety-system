<<<<<<< HEAD
# Bus Safety System

A real-time AI-powered safety system for buses that eliminates blind spots and detects driver drowsiness. Built with computer vision, edge computing, and sensor fusion for hackathon demonstrations.

##MVP Features

- **360° Blind Spot Detection**: Real-time object detection (pedestrians, vehicles, cyclists) using YOLOv8
- **Driver Drowsiness Detection**: Advanced head pose estimation + EAR (Eye Aspect Ratio) monitoring with baseline calibration
- **Head Tilt Detection**: Detects driver head tilt with temporal smoothing and adaptive calibration
- **Bird's Eye View Map**: Top-down visualization of objects around the bus using homography transformation
- **Emergency Alerts**: Automatic sound alerts and visual warnings when driver is drowsy or distracted
- **Live Dashboard**: Real-time Streamlit web dashboard showing BEV map, camera feeds, and alerts
- **Multi-Camera Support**: Supports USB webcams and phone cameras via IP streaming

## Tech Stack

- **Computer Vision**: YOLOv8, OpenCV, MediaPipe (optional fallback)
- **AI/ML**: Ultralytics YOLO, ByteTrack tracking
- **Edge Computing**: Real-time processing on laptop/edge device
- **Web Dashboard**: Streamlit
- **Hardware Integration**: Arduino support for physical brake mock

## Hardware Setup

- **4-6 webcams** (phone cameras via DroidCam/IP Webcam)
  - Front camera
  - Rear camera
  - Left mirror camera
  - Right mirror camera


=======
# bus-safety-system
>>>>>>> f0e5ff67ce73e4eb6903774c6817ca6b8739a3ee
