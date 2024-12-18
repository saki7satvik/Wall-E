import torch
import cv2
import numpy as np

# Load the PyTorch model
model_path = 'path_to_your_model.pt'  # Replace with your .pt file path
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define the class names corresponding to the model's output
class_names = ["Class1", "Class2", "Class3"]  # Replace with your actual class names

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to the input size expected by the model
    frame_tensor = torch.tensor(resized_frame / 255.0, dtype=torch.float32)  # Normalize
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Change shape to [1, 3, H, W]
    return frame_tensor

# OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera
if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break

    # Preprocess the frame
    input_tensor = preprocess_frame(frame)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = probabilities.topk(1)

    # Get the class name and probability
    class_name = class_names[top_class.item()]
    confidence = top_prob.item()

    # Draw a bounding box and label on the frame
    cv2.rectangle(frame, (10, 10), (300, 70), (0, 255, 0), -1)  # Background for text
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Plant Species Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
