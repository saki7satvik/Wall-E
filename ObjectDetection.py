import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv11 model with a custom configuration
model = YOLO('yolo11.yaml')  # Replace with the correct path to your YAML file

def detect_objects(image):
    """
    Detect objects in the image using the pretrained YOLOv11 model.
    Args:
        image (numpy.ndarray): The input image.
    Returns:
        numpy.ndarray: The image with bounding boxes and labels drawn.
    """
    # Perform inference on the image
    results = model.predict(image, save=False, conf=0.25)  # Adjust confidence threshold if needed
    
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates and class info
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_idx = int(box.cls)
            confidence = float(box.conf)  # Extract confidence
            
            # Get class name from the model's metadata
            class_name = model.names[class_idx]
            
            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, f'{class_name} {confidence:.2f}', 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
            )
    return image

def main():
    # Initialize video capture (use webcam or a video file)
    cap = cv2.VideoCapture(0)  # Replace '0' with video path for file input
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Process frame for object detection
        detected_frame = detect_objects(frame)

        # Display the processed frame
        cv2.imshow('Object Detection', detected_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
