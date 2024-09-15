import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load LBPH face recognizer and the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load OpenCV pre-trained face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the label dictionary from the saved labels.npy file
label_dict = np.load('labels.npy', allow_pickle=True).item()

# Route to display the front-end page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle face detection and recognition
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    img_data = data['image'].split(',')[1]  # Get base64 image string
    img_bytes = base64.b64decode(img_data)  # Decode the base64 string
    
    # Convert image bytes to numpy array and read as an OpenCV image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Draw a green rectangle around each face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Recognize the face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Confidence threshold
        threshold = 50  # Adjust this value based on your testing
        
        if confidence < threshold:
            # Get the name corresponding to the recognized ID from the label dictionary
            name = next((key for key, value in label_dict.items() if value == id), "Unknown")
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        # Display the name and confidence on the image
        cv2.putText(img, f"{name} {confidence_text}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Encode the image with rectangles and names as base64
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'success': True, 'image': jpg_as_text})

if __name__ == '__main__':
    app.run(debug=True)
