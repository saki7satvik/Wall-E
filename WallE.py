import serial
import threading
import time

class WallE:
    def __init__(self, port1):
        self.port1 = port1
        self.arduino_head = serial.Serial(port=self.port1, baudrate=9600, timeout=1)

    def move_up_down(self, servo, angle):
        # Ensure angle is an integer
        angle = int(angle)
        command = f"{servo}:{angle}\n"
        self.arduino_head.write(command.encode())
        time.sleep(2)
        print(f"Servo {servo} moved to {angle} degrees")

    def move_left_right(self, servo, angle):
        # Ensure angle is an integer
        angle = int(angle)
        command = f"{servo}:{angle}\n"
        self.arduino_head.write(command.encode())
        time.sleep(2)
        print(f"Servo {servo} moved to {angle} degrees")

    def initialise_all(self):
        # Use move_up_down and move_left_right functions for initialization
        thread1 = threading.Thread(target=self.move_up_down, args=('B', 45))
        thread2 = threading.Thread(target=self.move_left_right, args=('T', 90))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        print("All threads completed")

    def move_forward(self):
        print(f"Moving forward with motors connected to ports {self.port1}")
    
    def move_backward(self):
        print(f"Moving backward with motors connected to ports {self.port1}")  


# Create an instance of the WallE class
walle = WallE("COM6")
walle.initialise_all()
