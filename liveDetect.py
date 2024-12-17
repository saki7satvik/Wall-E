import cv2 as cv
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("satvik_detection1.pt")

# Input image path
input_image_path = "captured_images/image_22.jpg"  # Replace with your image path

# Load the image
image = cv.imread(input_image_path)
if image is None:
    print("Error: Unable to load image. Check the file path.")
    exit()

# Perform detection
results = model(image)

# Iterate through detection results
for i in results:
    for box in i.boxes:
        x, y, w, h = box.xywh[0]  # Extract bounding box coordinates
        class_id = int(box.cls[0])  # Extract class ID
        confidence = box.conf[0]  # Extract confidence score

        # Calculate top-left and bottom-right coordinates
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))

        # Draw the bounding box
        cv.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

        # Annotate the class name and confidence score
        class_name = model.names[class_id]
        text = f"{class_name} ({confidence:.2f})"
        cv.putText(image, text, (top_left[0], top_left[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Output image path
output_image_path = "output_image.jpg"

# Save the annotated image
cv.imwrite(output_image_path, image)
print(f"Output saved at {output_image_path}")

# Display the result
cv.imshow("Detected Objects", image)
cv.waitKey(0)
cv.destroyAllWindows()
