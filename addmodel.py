from ultralytics import YOLO

# Load the model with the specific configuration YAML
model = YOLO('yolov8n.yaml')  # Replace with your architecture YAML

# Transfer pre-trained weights if available (use 'yolov8n.pt' or your custom weights)
model.load('yolov8n.pt')  # Or specify another pretrained weights file

# Now you can use the model for inference, fine-tuning, or further training
# Load dataset YAML
# dataset_yaml = 'stapler_data.yaml'  # Path to your dataset configuration file

# Continue training with the dataset
# model.train(data=dataset_yaml, epochs=10, imgsz=640)
