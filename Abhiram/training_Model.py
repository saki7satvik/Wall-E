# Install required packages
# pip install ultralytics opencv-python face_recognition numpy

import cv2
import os
import numpy as np
from ultralytics import YOLO
import face_recognition

# Initialize YOLO face detection model
face_detector = YOLO('C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face_detection.pt')

# Path to your faces directory (should contain subdirectories named after people)
faces_dir = "C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face"

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load and encode known faces
for person_name in os.listdir(faces_dir):
    person_dir = os.path.join(faces_dir, person_name)
    if os.path.isdir(person_dir):
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # Detect face locations in the image
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Detect faces using YOLO
    results = face_detector.predict(frame, conf=0.7)
    
    # Get YOLO detection results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Process each detected face
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        label = face_detector.names[class_id]
        
        if label == "face" and confidence > 0.5:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract face region
            face_image = frame[y1:y2, x1:x2]
            
            # Convert to RGB format (required by face_recognition)
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if face_encodings:
                # Compare with known faces
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()