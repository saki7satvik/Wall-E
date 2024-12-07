import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov11n.pt" )


# Open the default camera
cam = cv.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))



while True:
    ret, frame = cam.read()

   
    
    if ret:
        results = model(frame)
        
            
            
        for i in results:
                
            for box in i.boxes:
                
                    
                for x, y, w, h in box.xywh:
                    print(f"x: {x}, y: {y}, w: {w}, h: {h}")
                   
                    
                    cv.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
                    class_name = model.names[int(box.cls)]
                    cv.putText(frame, class_name, (int(x - w / 2), int(y - h / 2) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv.imshow("Frame", frame)
                    
                    
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()

cv.destroyAllWindows()