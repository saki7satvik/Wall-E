import serial
from vosk import Model, KaldiRecognizer
import pyaudio
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

class BasicFunction:
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

    def send_command(self, command, description):
        """Sends a command to the Arduino."""
        if self.arduino:
            try:
                self.arduino.write(f"{command}\n".encode())
                print(description)
            except serial.SerialException as e:
                print(f"Failed to send command '{command}': {e}")
        else:
            print("Arduino is not connected.")

    def initialize_all(self):
        self.send_command("init", "initalized all")

    def look_up(self):
        self.send_command("look_up", "Top servo moved to look up.")

    def look_down(self):
        self.send_command("look_down", "Top servo moved to look down.")

    def look_left(self):
        self.send_command("turn_left", "Bottom servo moved to look left.")

    def look_right(self):
        self.send_command("turn_right", "Bottom servo moved to look right.")

    def go_right(self):
        self.send_command("go_right", "Robot moved to the right.")

    def go_left(self):
        self.send_command("go_left", "Robot moved to the left.")

    def go_front(self):
        self.send_command("go_front", "Robot moved to the front.")

    def go_back(self):
        self.send_command("go_back", "Robot moved to the back.")

    def dance(self):
        self.send_command("dance", "Robot is dancing.")

    def close_connection(self):
        """Closes the connection to the Arduino."""
        if self.arduino:
            try:
                self.arduino.close()
                print("Arduino connection closed.")
            except serial.SerialException as e:
                print(f"Failed to close Arduino connection: {e}")

class MLModel:
    def __init__(self, data):
        self.data = data
        self.df = pd.read_csv(data)

    def clean_text(self):
        def preprocess_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            return text
        self.df['voice_data'] = self.df['voice_data'].apply(preprocess_text)

    def vectorize(self):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.df['voice_data'])
        self.y = self.df['actual_command']

    def train_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        print("ML Model trained successfully.")

    def predict(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())  # Preprocess the text
        vectorized_text = self.vectorizer.transform([text])
        return self.model.predict(vectorized_text)[0]


# Main Program
if __name__ == "__main__":
    # Initialize the Arduino controller
    controller = BasicFunction(port="COM6")
    controller.initialize_connection()

    # Initialize the Vosk model for speech recognition
    model = Model("vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)

    # Initialize and train the ML model
    ml_model = MLModel(data="voice recognition model/updated_file.csv")
    ml_model.clean_text()
    ml_model.vectorize()
    ml_model.train_model()

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    print("Speak into the microphone...")

    try:
        while True:
            # Read audio stream
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                command = result.get("text", "")
                print(f"Recognized command: {command}")

                # Pass the command to the ML model and get the output
                ml_output = ml_model.predict(command)
                print(f"ML Model output: {ml_output}")

                # Map ML model output to Arduino actions
                if ml_output == "look up":
                    controller.look_up()
                elif ml_output == "look down":
                    controller.look_down()
                elif ml_output == "turn left":
                    controller.look_left()
                elif ml_output == "turn right":
                    controller.look_right()
                elif ml_output == "go right":
                    controller.go_right()
                elif ml_output == "go left":
                    controller.go_left()
                elif ml_output == "go front":
                    controller.go_front()
                elif ml_output == "go back":
                    controller.go_back()
                elif ml_output == "dance":
                    controller.dance()  
                else:
                    print("Invalid command. Please try again.")
                # Add more conditions based on your ML model's output
    except KeyboardInterrupt:
        print("Terminating program...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        controller.close_connection()