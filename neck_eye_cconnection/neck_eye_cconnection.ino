#include <Servo.h>

Servo topServo;     // Top servo
Servo bottomServo;  // Bottom servo

const int topServoPin = 7;
const int bottomServoPin = 8;

void setup() {
  Serial.begin(9600);
  topServo.attach(topServoPin);
  bottomServo.attach(bottomServoPin);
  
  // Set initial positions
  topServo.write(90);  // Move to 90 degrees
  bottomServo.write(90);  // Move to 90 degrees
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Read the entire command string
    if (input.startsWith("T:")) {
      int angle = input.substring(2).toInt();  // Extract the angle
      topServo.write(angle);  // Move top servo
      Serial.print("Top Servo moved to: ");
      Serial.println(angle);
    }
    else if (input.startsWith("B:")) {
      int angle = input.substring(2).toInt();  // Extract the angle
      bottomServo.write(angle);  // Move bottom servo
      Serial.print("Bottom Servo moved to: ");
      Serial.println(angle);
    }
  }
}

void init_servo(){
  topServo.write(135);
  bottomServo(135);
  delay(500);
  topServo.write(45);
  bottomServo(45);
  delay(500);
  topServo.write(90);
  bottomServo(90)
  delay(500);
}
