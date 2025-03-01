from deepface import DeepFace
import numpy as np
import cv2
import os

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

def get_embedding(image_path):
    # Extract facial embedding
    result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
    return result[0]['embedding'] if result else None

def compare_faces(image1, image2):
    emb1 = get_embedding(image1)
    emb2 = get_embedding(image2)
    
    if emb1 is None or emb2 is None:
        print("Could not extract embeddings for one or both images.")
        return None
    
    distance = euclidean_distance(emb1, emb2)
    print(f"Euclidean Distance: {distance}")
    
    # Lower distance means higher similarity
    if distance < 8:  # Adjust threshold based on your dataset
        print("Faces are similar!")
    else:
        print("Faces are different.")
    
    return distance

# Example usage
image1 = "./assets/satvik/1.png"
# image2 = "./assets/satvik/3.png"
dist = 0


def get_image_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_folder = "C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face/satvik"
image_files = get_image_files(image_folder)

image_folder_2 = "C:/Users/Abhiram/OneDrive/Desktop/HUMANOID/Abhiram/face/abhiram"
image_files_2 = get_image_files(image_folder_2)

previous_image = None

# for i, current_image in enumerate(image_files, start=1):
#     if previous_image is not None:
#         print(f"Comparing image {i-1} and image {i}")
#         dist += compare_faces(previous_image, current_image)
#     previous_image = current_image

for i, (image1, image2) in enumerate(zip(image_files, image_files_2), start=1):
    print(f"Comparing image {i} from folder 1 and image {i} from folder 2")
    dist += compare_faces(image1, image2)




print(dist / len(image_files))

for i in range(1, 21):
    print(i)
    image2 = f"./assets/satvik/{i}.png"
    dist += compare_faces(image1, image2)

print(dist/20)