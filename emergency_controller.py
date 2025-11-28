"""
Emergency Controller Module
Handles alerts and emergency actions when dangerous situations are detected
"""

import pygame
import time
from typing import Optional
import serial
import serial.tools.list_ports


class EmergencyController:
    def __init__(self, enable_sound: bool = True, enable_visual: bool = True,
                 enable_brake_mock: bool = False, arduino_port: Optional[str] = None,
                 brake_baudrate: int = 9600):
        """
        Initialize the emergency controller.
        
        Args:
            enable_sound: Enable sound alerts
            enable_visual: Enable visual alerts (handled by main system)
            enable_brake_mock: Enable Arduino brake mock
            arduino_port: Serial port for Arduino (None = auto-detect)
            brake_baudrate: Baud rate for Arduino communication
        """
        self.enable_sound = enable_sound
        self.enable_visual = enable_visual
        self.enable_brake_mock = enable_brake_mock
        
        # Initialize pygame for sound
        if enable_sound:
            try:
                pygame.mixer.init()
                # Create a simple alert sound programmatically
                self._create_alert_sound()
            except Exception as e:
                print(f"Warning: Could not initialize pygame mixer: {e}")
                self.enable_sound = False
        
        # Initialize Arduino connection
        self.arduino = None
        if enable_brake_mock:
            try:
                port = arduino_port or self._find_arduino_port()
                if port:
                    self.arduino = serial.Serial(port, brake_baudrate, timeout=1)
                    time.sleep(2)  # Wait for Arduino to initialize
                    print(f"Connected to Arduino on {port}")
                else:
                    print("Warning: Arduino not found, brake mock disabled")
                    self.enable_brake_mock = False
            except Exception as e:
                print(f"Warning: Could not connect to Arduino: {e}")
                self.enable_brake_mock = False
        
        # Alert state
        self.last_alert_time = 0
        self.alert_cooldown = 1.0  # Minimum seconds between alerts
        
    def _create_alert_sound(self):
        """Create a simple alert sound using pygame."""
        try:
            # Generate a louder, more noticeable beep sound (multiple beeps)
            sample_rate = 22050
            duration = 0.3  # Shorter beeps
            frequency = 1000  # Higher frequency for more attention
            
            import numpy as np
            # Create 3 beeps
            total_frames = int(sample_rate * duration * 3)  # 3 beeps
            arr = np.zeros((total_frames, 2), dtype=np.int16)
            max_sample = 2**(16 - 1) - 1
            
            beep_frames = int(sample_rate * duration)
            for beep_num in range(3):
                start_frame = beep_num * beep_frames * 2  # Space between beeps
                for i in range(beep_frames):
                    if start_frame + i < total_frames:
                        wave = max_sample * 0.8 * np.sin(2 * np.pi * frequency * i / sample_rate)
                        arr[start_frame + i][0] = int(wave)
                        arr[start_frame + i][1] = int(wave)
            
            self.alert_sound = pygame.sndarray.make_sound(arr)
            print("✓ Alert sound initialized")
        except Exception as e:
            print(f"Warning: Could not create alert sound: {e}")
            self.enable_sound = False
    
    def _find_arduino_port(self) -> Optional[str]:
        """Try to find Arduino port automatically."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Common Arduino identifiers
            if 'arduino' in port.description.lower() or 'usb' in port.description.lower():
                return port.device
        return None
    
    def trigger_alert(self, alert_type: str = "drowsiness", severity: str = "high"):
        """
        Trigger an emergency alert.
        
        Args:
            alert_type: Type of alert ("drowsiness", "obstacle", "collision")
            severity: Severity level ("low", "medium", "high")
        """
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        # Sound alert
        if self.enable_sound:
            try:
                if hasattr(self, 'alert_sound'):
                    # Play sound multiple times for emphasis
                    self.alert_sound.play()
                    # Also play system beep as backup
                    print("\a\a\a")  # Triple beep
                else:
                    # Fallback: system beep (multiple times)
                    print("\a\a\a")  # Triple beep
            except Exception as e:
                print(f"Error playing alert sound: {e}")
                # Fallback to system beep
                print("\a\a\a")
        
        # Arduino brake mock
        if self.enable_brake_mock and self.arduino:
            try:
                self.arduino.write(b"BRAKE\n")
                print("Sent BRAKE command to Arduino")
            except Exception as e:
                print(f"Error sending brake command: {e}")
        
        print(f"ALERT TRIGGERED: {alert_type} ({severity} severity)")
    
    def trigger_emergency_brake(self):
        """Trigger emergency braking action."""
        self.trigger_alert("emergency_brake", "high")
    
    def trigger_drowsiness_alert(self):
        """Trigger drowsiness alert."""
        self.trigger_alert("drowsiness", "high")
    
    def trigger_obstacle_alert(self):
        """Trigger obstacle proximity alert."""
        self.trigger_alert("obstacle", "medium")
    
    def cleanup(self):
        """Clean up resources."""
        if self.arduino:
            try:
                self.arduino.close()
            except:
                pass
        
        if self.enable_sound:
            try:
                pygame.mixer.quit()
            except:
                pass

