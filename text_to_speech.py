from vosk import Model, KaldiRecognizer
import json
import pyaudio
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Load the Vosk model
model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)


# Preprocess text for consistency
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# Load and train the command recognition model
def train_command_model():
    df = pd.read_csv('D:/wallE/human detection/voice recognition model/updated_file.csv').dropna()
    df['voice_data'] = df['voice_data'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['voice_data'])
    y = df['actual_command']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    command_model = LogisticRegression()
    command_model.fit(X_train, y_train)

    return command_model, vectorizer


# Train the model and load the vectorizer
command_model, vectorizer = train_command_model()

# Configure audio stream
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

print("Speak into the microphone...")

try:
    while True:
        # Capture audio data
        data = stream.read(4096, exception_on_overflow=False)
        
        # Recognize speech from the audio
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            recognized_text = result['text']

            # Preprocess the recognized text
            processed_text = preprocess_text(recognized_text)

            # Vectorize the processed text
            input_vector = vectorizer.transform([processed_text])

            # Predict the command
            predicted_command = command_model.predict(input_vector)[0]

            print(f"Recognized text: {recognized_text}")
            print(f"Predicted command: {predicted_command}")

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Properly close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
