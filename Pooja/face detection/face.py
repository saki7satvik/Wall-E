import cv2
from ultralytics import YOLO
import mediapipe as mp
import sqlite3
from datetime import datetime

model = YOLO("satvik_detection1.pt")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
conn = sqlite3.connect('detection_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS detections
             (timestamp TEXT, class_name TEXT, confidence REAL, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER)''')
conn.commit()
def store_detection(class_name, confidence, x1, y1, x2, y2):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence_value = confidence.numpy().item()  
    confidence_value = float(confidence_value)

    c.execute("INSERT INTO detections (timestamp, class_name, confidence, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (timestamp, class_name, confidence_value, x1, y1, x2, y2))
    conn.commit()

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    results = model(frame, verbose=False)
    face_detected = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if confidence > 0.5 and class_id == 0: 
                face_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Your Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                store_detection("satvik_detection1", confidence, x1, y1, x2, y2)

    if face_detected:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                if abs(thumb_tip.y - index_tip.y) > 0.05 and abs(middle_tip.y - wrist.y) > 0.2:
                    cv2.putText(frame, "Open Palm Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                elif thumb_tip.y < wrist.y and abs(index_tip.y - wrist.y) > 0.1:
                    cv2.putText(frame, "Thumbs Up!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif thumb_tip.y > wrist.y and abs(index_tip.y - wrist.y) > 0.1:
                    cv2.putText(frame, "Thumbs Down!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face and Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
conn.close()