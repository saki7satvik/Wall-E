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
  topServo.write(90);     // Move to 90 degrees
  bottomServo.write(90);  // Move to 90 degrees
  Serial.println("Servos initialized and ready.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Read the entire command string

    // Handle "init" command to initialize servos
    if (input.startsWith("init")) {
      init_servo();
      Serial.println("Initialization complete.");
    }
    // Handle commands for top servo (T)
    else if (input.startsWith("T:")) {
      int angle = input.substring(2).toInt();  // Extract the angle
      if (angle >= 0 && angle <= 180) {
        topServo.write(angle);  // Move top servo
        Serial.print("Top servo moved to: ");
        Serial.println(angle);
      } else {
        Serial.println("Invalid angle for top servo. Use values between 0 and 180.");
      }
    }
    // Handle commands for bottom servo (B)
    else if (input.startsWith("B:")) {
      int angle = input.substring(2).toInt();  // Extract the angle
      if (angle >= 0 && angle <= 180) {
        bottomServo.write(angle);  // Move bottom servo
        Serial.print("Bottom servo moved to: ");
        Serial.println(angle);
      } else {
        Serial.println("Invalid angle for bottom servo. Use values between 0 and 180.");
      }
    } else {
      Serial.println("Invalid command. Use 'init', 'T:<angle>', or 'B:<angle>'.");
    }
  }
}

void init_servo() {
  topServo.write(135);
  bottomServo.write(135);
  delay(500);
  topServo.write(45);
  bottomServo.write(45);
  delay(500);
  topServo.write(90);
  bottomServo.write(90);
  delay(500);
}
