import cv2
import os

def capture_images(save_dir, num_images=20, camera_index=0):
    """
    Captures multiple images using OpenCV and saves them in a directory.
    
    Args:
        save_dir (str): Directory to save the captured images.
        num_images (int): Number of images to capture.
        camera_index (int): Index of the camera to use.
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return
    
    print("Press 'Space' to capture an image and 'q' to quit.")
    captured_images = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the frame
        cv2.imshow("Capture Images", frame)
        
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, quit
        if key == ord('q'):
            print("Exiting capture.")
            break

        # If 'Space' is pressed, save the image
        if key == ord(' '):
            filename = os.path.join(save_dir, f'image_{captured_images + 1}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            captured_images += 1

            # Stop if the desired number of images are captured
            if captured_images >= num_images:
                print("Captured all images.")
                break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Directory to save images
output_directory = "captured_images"
# Number of images to capture
number_of_images = 20

# Call the function
capture_images(output_directory, number_of_images)
