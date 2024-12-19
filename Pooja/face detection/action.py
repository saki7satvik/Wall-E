import cv2
import mediapipe as mp
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import math
import os

# model_path = "Pooja\face detection\satvik_detection1.pt"

model = YOLO("C:/Users/pmetr/Projects/Projects/Wall-E/Pooja/face detection/satvik_detection1.pt")  # Load the custom YOLO model
 # Load the custom YOLO model

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize SQLite database
conn = sqlite3.connect('detection_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS detections
             (timestamp TEXT, class_name TEXT, confidence REAL, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER)''')
conn.commit()

def store_detection(class_name, confidence, x1, y1, x2, y2):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO detections (timestamp, class_name, confidence, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (timestamp, class_name, float(confidence.numpy().item()), x1, y1, x2, y2))
    conn.commit()

# Gesture detection helper functions
def is_open_palm(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    return all(hand_landmarks.landmark[tip].y < wrist.y for tip in tips)

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < thumb_mcp.y and index_tip.y > thumb_mcp.y

def is_thumbs_down(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y > thumb_mcp.y and index_tip.y < thumb_mcp.y

def is_middle_finger_raised(hand_landmarks):
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return (middle_tip.y < middle_pip.y and
            all(finger_tip.y > middle_pip.y for finger_tip in [index_tip, ring_tip, pinky_tip]))

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = model(frame, verbose=False)  # Object detection using YOLO model
    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    # Face detection (based on YOLO output)
    for result in face_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if confidence > 0.5:  # Only detect if confidence is greater than threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Object Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                store_detection("satvik_detection1", confidence, x1, y1, x2, y2)

    # Hand gesture detection
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_open_palm(hand_landmarks):
                cv2.putText(frame, "Open Palm Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_thumbs_up(hand_landmarks):
                cv2.putText(frame, "Thumbs Up!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_thumbs_down(hand_landmarks):
                cv2.putText(frame, "Thumbs Down!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_middle_finger_raised(hand_landmarks):
                cv2.putText(frame, "Middle Finger Detected!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Pose detection for open arms
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        lw, rw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        ls, rs = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hand_dist = math.sqrt((rw.x - lw.x)**2 + (rw.y - lw.y)**2)
        shoulder_dist = math.sqrt((rs.x - ls.x)**2 + (rs.y - ls.y)**2)
        if hand_dist > 1.5 * shoulder_dist:
            cv2.putText(frame, "Open Arms Detected!", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face and Gesture Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()