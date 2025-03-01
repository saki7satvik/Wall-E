import cv2
from ultralytics import YOLO
from time import sleep

class FaceDetection:
    def __init__(self, model_path, cam1=0, cam2=None, save_limit=10):
        self.model = YOLO(model_path)
        self.cam1 = cv2.VideoCapture(cam1)
        self.save_count = 0  # Counter for saved images
        self.save_limit = save_limit  # Max number of images to save
        if cam2:
            self.cam2 = cv2.VideoCapture(cam2)

    def save_detected_face(self, frame, x1, y1, x2, y2):
        """Crops and saves the detected face."""
        if self.save_count >= self.save_limit:
            return  # Stop saving after reaching the limit
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size != 0:
            filename = f"C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face/detected_face_{self.save_count + 1}.jpg"
            cv2.imwrite(filename, face_crop)
            self.save_count += 1
            print(f"Saved: {filename}")
            sleep(2)

    def detect(self):
        while self.save_count < self.save_limit:
            ret, frame = self.cam1.read()
            if not ret:
                break

            results = self.model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    label = self.model.names[class_id]

                    if label == "face" and confidence > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Call function to save face
                        self.save_detected_face(frame, x1, y1, x2, y2)
            
            

           

            


# Run the face detection with a limit of 10 saved images
face_detector = FaceDetection("C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face_detection.pt", cam1=0, save_limit=20)
face_detector.detect()