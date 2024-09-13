import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

image_path = 'TrainingImage/'
faces = []
labels = []

for filename in os.listdir(image_path):
    img_path = os.path.join(image_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    label = int(filename.split('_')[0])
    faces.append(img)
    labels.append(label)

recognizer.train(faces, np.array(labels))
recognizer.save('models/trained_model.yml')
