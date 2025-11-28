"""
Streamlit Dashboard for Real-time Bus Safety System Monitoring
"""

import streamlit as st
import cv2
import numpy as np
import time
import yaml
import os
from PIL import Image

# Import system components
from main import BusSafetySystem

# Page configuration
st.set_page_config(
    page_title="Bus Safety System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #ff4444;
        color: white;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
    }
    .alert-success {
        background-color: #00aa00;
        color: white;
    }
    .metric-card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = time.time()
if 'drowsiness_alert_time' not in st.session_state:
    st.session_state.drowsiness_alert_time = 0
if 'head_tilt_alert_time' not in st.session_state:
    st.session_state.head_tilt_alert_time = 0

def initialize_system():
    """Initialize the bus safety system."""
    config_path = st.sidebar.text_input("Config file", value="config.yaml")
    
    if os.path.exists(config_path):
        try:
            system = BusSafetySystem(config_path=config_path)
            st.session_state.system = system
            return True
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return False
    else:
        st.error(f"Config file not found: {config_path}")
        return False

def convert_frame_to_rgb(frame):
    """Convert BGR frame to RGB for display."""
    if frame is None:
        return None
    try:
        # Validate frame
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        
        # Check if frame has valid dimensions
        if len(frame.shape) < 2:
            return None
        
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2:
            # Grayscale frame, convert to RGB
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            return frame
    except Exception as e:
        print(f"Error converting frame: {e}")
        return None

# Sidebar
st.sidebar.title(" Control Panel")

# System control
st.sidebar.subheader("System Control")
if st.sidebar.button("Initialize System"):
    if initialize_system():
        st.sidebar.success("System initialized!")

if st.session_state.system is not None:
    if not st.session_state.system_running:
        if st.sidebar.button("▶️ Start System"):
            st.session_state.system.start()
            st.session_state.system_running = True
            st.sidebar.success("System started!")
            st.rerun()
    else:
        if st.sidebar.button("⏹️ Stop System"):
            st.session_state.system.stop()
            st.session_state.system_running = False
            st.sidebar.warning("System stopped!")
            st.rerun()

# Configuration display
st.sidebar.subheader("Configuration")
if st.session_state.system is not None:
    config = st.session_state.system.config
    st.sidebar.json({
        "Model": config['detection']['model'],
        "EAR Threshold": config['drowsiness']['ear_threshold'],
        "Danger Zone": f"{config['detection']['danger_zone_radius']}m"
    })

# Main content
st.markdown('<h1 class="main-header"> Bus Safety System</h1>', unsafe_allow_html=True)

# Status indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_color = "🟢" if st.session_state.system_running else "🔴"
    st.metric("System Status", status_color)

with col2:
    if st.session_state.system_running and st.session_state.system.latest_drowsiness:
        drowsy_status = " DROWSY" if st.session_state.system.latest_drowsiness['is_drowsy'] else " AWAKE"
        st.metric("Driver Status", drowsy_status)
    else:
        st.metric("Driver Status", "N/A")

with col3:
    if st.session_state.system_running:
        total_detections = sum(
            len(det.get("boxes", [])) if det else 0
            for det in st.session_state.system.latest_detections.values()
        )
        st.metric("Total Detections", total_detections)
    else:
        st.metric("Total Detections", 0)

with col4:
    if st.session_state.system_running:
        dangerous_count = sum(
            len(st.session_state.system.blind_spot_detector.get_dangerous_objects(det))
            for det in st.session_state.system.latest_detections.values()
            if det is not None
        )
        st.metric("Dangerous Objects", dangerous_count)
    else:
        st.metric("Dangerous Objects", 0)

# Alert section
st.markdown("---")
st.subheader("Alerts & Warnings")

# Load alert duration from config
alert_duration = 5.0  # Default duration
try:
    if st.session_state.system and hasattr(st.session_state.system, 'config'):
        alert_duration = st.session_state.system.config.get('dashboard', {}).get('alert_duration', 5.0)
except Exception:
    pass  # Use default if config loading fails

if st.session_state.system_running and st.session_state.system.latest_drowsiness:
    drowsiness = st.session_state.system.latest_drowsiness
    current_time = time.time()
    
    # Update alert timestamps when alerts are triggered
    if drowsiness['is_drowsy']:
        st.session_state.drowsiness_alert_time = current_time
    if drowsiness.get('head_tilted', False):
        st.session_state.head_tilt_alert_time = current_time
    
    # Check if alerts should still be visible (within duration window)
    drowsiness_alert_active = (current_time - st.session_state.drowsiness_alert_time) < alert_duration
    head_tilt_alert_active = (current_time - st.session_state.head_tilt_alert_time) < alert_duration
    
    # Show alerts based on current state or timer
    if drowsiness['is_drowsy'] or drowsiness_alert_active:
        st.markdown(
            '<div class="alert-box alert-danger">DRIVER DROWSY - EMERGENCY BRAKE ACTIVATED!</div>',
            unsafe_allow_html=True
        )
    elif drowsiness.get('head_tilted', False) or head_tilt_alert_active:
        max_tilt = max(abs(drowsiness.get('head_pitch', 0)),
                      abs(drowsiness.get('head_yaw', 0)),
                      abs(drowsiness.get('head_roll', 0)))
        st.markdown(
            f'<div class="alert-box alert-warning">⚠️ HEAD TILT DETECTED - Angle: {max_tilt:.1f}° - ALERT TRIGGERED!</div>',
            unsafe_allow_html=True
        )
    elif drowsiness['face_detected']:
        st.markdown(
            '<div class="alert-box alert-success"> Driver is awake and alert</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="alert-box alert-warning">⚠️ No face detected - check driver camera</div>',
            unsafe_allow_html=True
        )
else:
    st.info("System not running. Click 'Start System' to begin monitoring.")

# Main display area
st.markdown("---")

if st.session_state.system_running:
    # Create two columns for main displays
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader(" Driver Monitoring")
        
        if st.session_state.system.latest_drowsiness:
            driver_frame = st.session_state.system.latest_drowsiness.get('annotated_frame')
            if driver_frame is not None:
                driver_rgb = convert_frame_to_rgb(driver_frame)
                if driver_rgb is not None:
                    try:
                        st.image(driver_rgb, channels="RGB", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying driver frame: {e}")
                        st.info("Camera feed temporarily unavailable")
                else:
                    st.warning("Invalid driver frame")
                
                # Drowsiness metrics
                drowsiness = st.session_state.system.latest_drowsiness
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                with col_metric1:
                    st.metric("EAR", f"{drowsiness['ear_avg']:.3f}")
                with col_metric2:
                    st.metric("PERCLOS", f"{drowsiness['perclos']:.2%}")
                with col_metric3:
                    st.metric("Closed Frames", f"{drowsiness['closed_eye_counter']}/{drowsiness['consecutive_frames_threshold']}")
                with col_metric4:
                    max_tilt = max(abs(drowsiness.get('head_pitch', 0)),
                                 abs(drowsiness.get('head_yaw', 0)),
                                 abs(drowsiness.get('head_roll', 0)))
                    tilt_status = "⚠️ TILTED" if drowsiness.get('head_tilted', False) else "✓ Normal"
                    st.metric("Head Tilt", f"{max_tilt:.1f}°")
                    st.caption(tilt_status)
            else:
                st.warning("No driver frame available")
        else:
            st.info("Waiting for driver camera feed...")
    
    with col_right:
        st.subheader("Bird's Eye View Map")
        
        if st.session_state.system.latest_bev_map is not None:
            bev_rgb = convert_frame_to_rgb(st.session_state.system.latest_bev_map)
            if bev_rgb is not None:
                try:
                    st.image(bev_rgb, channels="RGB", use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying BEV map: {e}")
                    st.info("BEV map temporarily unavailable")
            else:
                st.warning("Invalid BEV map")
        else:
            st.info("Generating BEV map...")
    
    # Camera feeds section
    st.markdown("---")
    st.subheader("📹 Camera Feeds")
    
    camera_cols = st.columns(4)
    camera_names = ["front", "rear", "left", "right"]
    
    for i, camera_name in enumerate(camera_names):
        with camera_cols[i]:
            st.write(f"**{camera_name.upper()}**")
            
            # Check if frame exists (don't require detection to show feed)
            frame = st.session_state.system.latest_frames.get(camera_name)
            detection = st.session_state.system.latest_detections.get(camera_name)
            
            if frame is not None:
                try:
                    # Show frame even if detection hasn't been processed yet
                    if detection:
                        annotated = st.session_state.system.blind_spot_detector.draw_detections(
                            frame, detection
                        )
                    else:
                        annotated = frame
                    
                    frame_rgb = convert_frame_to_rgb(annotated)
                    if frame_rgb is not None:
                        st.image(frame_rgb, channels="RGB", use_container_width=True)
                    else:
                        st.warning("Invalid frame data")
                except Exception as e:
                    st.error(f"Error displaying {camera_name} camera: {e}")
                    st.info("Camera feed temporarily unavailable")
                
                # Detection count
                if detection:
                    count = len(detection.get("boxes", []))
                    st.caption(f"Detections: {count}")
                else:
                    st.caption("Processing...")
            else:
                # Check if camera is configured
                camera_config = st.session_state.system.config['cameras']
                if camera_name in camera_config:
                    source = camera_config[camera_name]
                    if isinstance(source, str):
                        st.warning(f"Phone camera not connected")
                        st.caption(f"URL: {source}")
                    else:
                        st.warning(f"Camera {source} not accessible")
                else:
                    st.info("Camera not configured")
    
    # Auto-refresh (with rate limiting to prevent 404s and unhandled promise rejections)
    current_time = time.time()
    min_refresh_interval = 0.033  # ~30 FPS max
    
    # Safely get last_refresh_time with default value
    last_refresh = st.session_state.get('last_refresh_time', current_time)
    st.session_state.last_refresh_time = last_refresh
    
    if current_time - last_refresh >= min_refresh_interval:
        st.session_state.last_refresh_time = current_time
        try:
            time.sleep(0.033)  # ~30 FPS
            st.rerun()
        except Exception:
            # Handle refresh errors gracefully - prevents unhandled promise rejections
            # Don't show error to user, just wait and retry
            time.sleep(0.1)
            try:
                st.rerun()
            except:
                pass  # Silently fail if rerun still fails
    else:
        # Rate limit - wait before next refresh
        sleep_time = min_refresh_interval - (current_time - last_refresh)
        if sleep_time > 0:
            time.sleep(sleep_time)
        try:
            st.rerun()
        except:
            pass
else:
    st.info(" Initialize and start the system to see live monitoring data.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>Bus Safety System </div>",
    unsafe_allow_html=True
)

