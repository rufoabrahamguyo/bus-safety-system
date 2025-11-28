/*
 * Arduino Brake Pedal Mock Controller
 * 
 * This sketch controls a servo motor to simulate brake pedal pressing
 * when receiving "BRAKE" command from Python via serial.
 * 
 * Hardware:
 * - Arduino Uno/Nano
 * - Servo motor (SG90 or similar)
 * - Optional: LED for visual feedback
 * 
 * Connections:
 * - Servo signal pin -> Digital Pin 9
 * - Servo VCC -> 5V
 * - Servo GND -> GND
 * - LED (optional) -> Digital Pin 13 with resistor
 * 
 * Usage:
 * 1. Upload this sketch to Arduino
 * 2. Connect servo to brake pedal mechanism
 * 3. Set enable_brake_mock: true in config.yaml
 * 4. System will send "BRAKE" command when emergency detected
 */

#include <Servo.h>

Servo brakeServo;

// Pin definitions
const int SERVO_PIN = 9;
const int LED_PIN = 13;

// Servo positions
const int SERVO_REST = 90;      // Rest position (no brake)
const int SERVO_BRAKE = 0;      // Brake position (pressed)
const int SERVO_DURATION = 500; // How long to hold brake (ms)

// State
bool brakeActive = false;
unsigned long brakeStartTime = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  Serial.setTimeout(100);
  
  // Initialize servo
  brakeServo.attach(SERVO_PIN);
  brakeServo.write(SERVO_REST);
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Wait for serial connection
  delay(1000);
  
  Serial.println("Brake Mock Controller Ready");
  Serial.println("Send 'BRAKE' to activate emergency brake");
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    
    if (command == "BRAKE") {
      activateBrake();
    } else if (command == "RESET") {
      resetBrake();
    }
  }
  
  // Auto-reset brake after duration
  if (brakeActive) {
    if (millis() - brakeStartTime > SERVO_DURATION) {
      resetBrake();
    }
  }
}

void activateBrake() {
  brakeActive = true;
  brakeStartTime = millis();
  
  // Move servo to brake position
  brakeServo.write(SERVO_BRAKE);
  
  // Turn on LED
  digitalWrite(LED_PIN, HIGH);
  
  Serial.println("BRAKE ACTIVATED!");
}

void resetBrake() {
  brakeActive = false;
  
  // Return servo to rest position
  brakeServo.write(SERVO_REST);
  
  // Turn off LED
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("Brake reset");
}

