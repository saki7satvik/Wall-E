import cv2
import torch
from ultralytics import YOLO  # Assuming you have a YOLOv5 model

# Load the pretrained YOLOv11 model
model = YOLO('yolo11l.pt')

# Pascal VOC classes
# VOC_CLASSES = [
#     "person", "bird", "cat", "cow", "dog", "horse", "sheep",
#     "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
#     "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"
# ]

# def detect_objects(image):
#     """
#     Detect objects in the image using the pretrained YOLOv11 model.
#     Args:
#         image (numpy.ndarray): The input image.
#     Returns:
#         numpy.ndarray: The image with bounding boxes and labels drawn.
#     """
#     results = model(image)
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()  # Ensure proper unpacking
#             class_idx = int(box.cls)
#             if class_idx < len(VOC_CLASSES):
#                 label = VOC_CLASSES[class_idx]
#                 confidence = float(box.conf)  # Convert tensor to float
#                 cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     return image

def main():
    # Read the image from file
    # image_path = 'image.png'
    # image = cv2.imread(image_path)

    # if image is None:
    #     print("Error: Could not read image.")
    #     return

    # # Detect objects in the image
    # detected_image = detect_objects(image)
    
    # # Display the image
    # cv2.imshow("Object Detection", detected_image)
    
    # # Wait for a key press and close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    classes = model.names
    print(model.names)

if __name__ == "__main__":
    main()