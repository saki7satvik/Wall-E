import serial

class ServoController:
    def __init__(self, port, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.arduino = None

    def initialize_connection(self):
        """Initializes the Arduino connection."""
        try:
            self.arduino = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
            print(f"Connected to Arduino on port {self.port}.")
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino on port {self.port}: {e}")
            self.arduino = None

    def look_up(self):
        """Moves the top servo to look up."""
        try:
            self.arduino.write(b"T:0\n")
            print("Top servo moved to look up.")
        except serial.SerialException as e:
            print(f"Failed to move top servo to look up: {e}")
            self.arduino = None   

    def send_command(self, servo, angle=None):
        """Sends a command to move the servos or initialize them."""
        if self.arduino is None:
            print("Arduino connection is not available.")
            return

        if servo == "init":
            command = "init\n"
        elif servo in ["T", "B"] and angle is not None:
            if not (0 <= angle <= 180):
                print("Invalid angle. Use a value between 0 and 180.")
                return
            command = f"{servo}:{angle}\n"
        else:
            print("Invalid command. Use 'init', 'T:<angle>', or 'B:<angle>'.")
            return

        self.arduino.write(command.encode())
        print(f"Command sent: {command.strip()}")

    def close_connection(self):
        """Closes the Arduino connection."""
        if self.arduino:
            self.arduino.close()
            print("Arduino connection closed.")


# Main Program
if __name__ == "__main__":
    # Create an instance of the ServoController
    controller = ServoController(port="COM6")

    # Initialize the Arduino connection
    controller.initialize_connection()

    try:
        while True:
            user_input = input("Enter command (e.g., 'T:90', 'B:45', or 'init'): ").strip()
            if user_input == "init":
                controller.send_command("init")
            elif user_input.startswith("T:") or user_input.startswith("B:"):
                try:
                    servo, angle = user_input.split(":")
                    angle = int(angle)
                    controller.send_command(servo, angle)
                except ValueError:
                    print("Invalid command format. Use 'T:<angle>' or 'B:<angle>'.")
            else:
                print("Invalid command. Use 'init', 'T:<angle>', or 'B:<angle>'.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        controller.close_connection()
