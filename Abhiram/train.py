from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt', 'yolov8m.pt' for larger models

# Train the model
model.train(
    data="C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face_dataset/dataset.yaml",  # Path to dataset.yaml
    epochs=100,           # Number of training epochs
    batch=16,             # Batch size
    imgsz=640,            # Image size
    name="yolov8_training",  # Experiment name
    project="runs/train"   # Directory to save results
)

# Validate the trained model
results = model.val()
print(results)

# Export the model to ONNX and TorchScript format for deployment
model.export(format="onnx")  # Save model as ONNX
model.export(format="torchscript")  # Save model as TorchScript

print("Training complete. Model exported successfully!")
