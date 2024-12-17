# import serial

# # Replace 'COM6' with the port your Arduino is connected to
# arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)

# def send_command(servo, angle):
#     if servo not in ['T', 'B']:
#         print("Invalid servo. Use 'T' for top or 'B' for bottom.")
#         return

#     if not (0 <= angle <= 180):
#         print("Invalid angle. Use a value between 0 and 180.")
#         return

#     command = f"{servo}:{angle}\n"
#     arduino.write(command.encode())
#     print(f"Command sent: {command.strip()}")

# while True:
#     try:
#         servo = input("Enter servo (T for top, B for bottom): ").strip().upper()
#         angle = int(input("Enter angle (0-180): ").strip())
#         send_command(servo, angle)
#     except ValueError:
#         print("Invalid input. Please enter a valid angle.")
#     except KeyboardInterrupt:
#         print("\nExiting...")
#         break


import serial

class ServoController:
    def __init__(self, port, baudrate=9600, timeout=1):
        """
        Initialize the ServoController class with the serial connection to the Arduino.
        """
        try:
            self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            print(f"Connected to Arduino on port {port}")
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.arduino = None

    def send_command(self, servo, angle):
        """
        Sends a command to move the servo to the specified angle.
        
        :param servo: 'T' for top servo, 'B' for bottom servo.
        :param angle: Angle between 0 and 180 degrees.
        """
        if servo not in ['T', 'B']:
            print("Invalid servo. Use 'T' for top or 'B' for bottom.")
            return

        if not (0 <= angle <= 180):
            print("Invalid angle. Use a value between 0 and 180.")
            return

        command = f"{servo}:{angle}\n"
        if self.arduino:
            self.arduino.write(command.encode())
            print(f"Command sent: {command.strip()}")
        else:
            print("Arduino connection not initialized.")

    def close_connection(self):
        """
        Closes the serial connection.
        """
        if self.arduino:
            self.arduino.close()
            print("Arduino connection closed.")


class ServoControlApp:
    def __init__(self, servo_controller):
        """
        Initialize the ServoControlApp with a ServoController instance.
        """
        self.servo_controller = servo_controller

    def run(self):
        """
        Runs the interactive CLI application for controlling servos.
        """
        print("Enter servo commands ('T' for top, 'B' for bottom) and angles (0-180).")
        print("Press Ctrl+C to exit.")

        while True:
            try:
                servo = input("Enter servo (T for top, B for bottom): ").strip().upper()
                angle = int(input("Enter angle (0-180): ").strip())
                self.servo_controller.send_command(servo, angle)
            except ValueError:
                print("Invalid input. Please enter a valid angle.")
            except KeyboardInterrupt:
                print("\nExiting...")
                self.servo_controller.close_connection()
                break


# Main code
if __name__ == "__main__":
    arduino_port = "COM6"  # Replace with your Arduino's port
    servo_controller = ServoController(port=arduino_port)
    app = ServoControlApp(servo_controller)
    app.run()
