import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to get the images and label data
def get_images_and_labels(path):
    face_samples = []
    ids = []
    label_dict = {}  # To map names to numeric IDs
    current_id = 0

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if os.path.isdir(folder_path):
            # Assign a numeric ID to each person (folder name)
            if folder_name not in label_dict:
                label_dict[folder_name] = current_id
                current_id += 1

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                # Open the image and convert it to grayscale
                img = Image.open(image_path).convert('L')
                img_np = np.array(img, 'uint8')

                # Detect faces in the image
                faces = detector.detectMultiScale(img_np)

                for (x, y, w, h) in faces:
                    face_samples.append(img_np[y:y+h, x:x+w])
                    ids.append(label_dict[folder_name])

    return face_samples, ids, label_dict

faces, ids, label_dict = get_images_and_labels(path)

# Convert ids to numpy array of int32 type
ids = np.array(ids, dtype=np.int32)

# Train the recognizer
recognizer.train(faces, ids)

# Save the trained model
recognizer.write('trainer.yml')

# Save label dictionary for future use (optional)
np.save('labels.npy', label_dict)

print("Model training complete. Label dictionary:", label_dict)
