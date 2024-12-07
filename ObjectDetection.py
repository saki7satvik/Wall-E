from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('stapler.pt')  # Use 'yolov5s.pt', 'yolov8n.pt' or your custom model

# Initialize video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)  # Use 'video.mp4' for video file

# Define class names (COCO classes)
class_names = model.names

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame)

    # results.xyxy[0] gives you the bounding boxes and related data
    # results.pandas().xywh gives you the bounding boxes as pandas DataFrame (alternative)
    for result in results:
        boxes = result.boxes  # bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            conf = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = f"{class_names[class_id]} {conf:.2f}"

            # Draw bounding boxes
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()