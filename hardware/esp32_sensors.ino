/*
 * ESP32 Sensor Hub for VisionMate Smart Glasses
 * Reads: Ultrasonic sensor (HC-SR04) + Push button
 * Sends data to Raspberry Pi via USB Serial
 * 
 * Connections:
 * - Ultrasonic TRIG -> GPIO 18
 * - Ultrasonic ECHO -> GPIO 19
 * - Button -> GPIO 0 (with pull-up)
 * - USB -> Raspberry Pi
 */

#define TRIG_PIN 18     // Ultrasonic trigger pin
#define ECHO_PIN 19     // Ultrasonic echo pin
#define BUTTON_PIN 0    // Push button pin (boot button can be used)

// State tracking
bool lastButtonState = HIGH;
unsigned long lastSendTime = 0;
const int SEND_INTERVAL = 100;  // Send data every 100ms (10Hz)

void setup() {
  // Initialize serial communication (USB to RPi)
  Serial.begin(115200);
  
  // Configure pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Internal pull-up resistor
  
  // Startup indicator
  Serial.println("ESP32:READY");
  delay(1000);
}

void loop() {
  unsigned long currentTime = millis();
  
  // Send sensor data at regular intervals
  if (currentTime - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = currentTime;
    
    // Read ultrasonic distance
    float distance = readUltrasonic();
    
    // Read button state
    bool buttonPressed = (digitalRead(BUTTON_PIN) == LOW);
    
    // Send data in compact format: "DIST:123.45,BTN:0"
    Serial.print("DIST:");
    Serial.print(distance, 2);  // 2 decimal places
    Serial.print(",BTN:");
    Serial.println(buttonPressed ? "1" : "0");
  }
  
  delay(10);  // Small delay to prevent overwhelming serial
}

float readUltrasonic() {
  // Trigger ultrasonic pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read echo pulse duration (timeout after 30ms)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  
  // Calculate distance in centimeters
  // Speed of sound = 343 m/s = 0.0343 cm/µs
  // Distance = (duration / 2) * 0.0343
  float distance = duration * 0.01715;
  
  // Validate range (HC-SR04: 2-400 cm)
  if (distance < 2 || distance > 400 || duration == 0) {
    return 999.0;  // Out of range indicator
  }
  
  return distance;
}
