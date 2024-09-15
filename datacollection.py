import cv2
import os

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def collect_face_data(name, num_images=30):
    # Create a directory for storing face data
    if not os.path.exists(f'dataset/{name}'):
        os.makedirs(f'dataset/{name}')
    
    cam = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"dataset/{name}/{count}.jpg", gray[y:y+h, x:x+w])  # Save the captured image
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Collecting Faces', img)

        if count >= num_images:
            break

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

collect_face_data('YourName')
