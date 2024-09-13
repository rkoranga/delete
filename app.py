from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

# Load the face recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')  # Your pre-trained model
cascadePath = "Cascades/haarcascade_frontalface_default.xml"  # Path to Haar Cascade
faceCascade = cv2.CascadeClassifier(cascadePath)

# Names associated with face IDs
names = ['None', 'Dharmaraj', 'Vikram']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Route to process the frame sent from the client
@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Receive the frame from the client
    file = request.files['frame']
    img_bytes = file.read()

    # Convert the image bytes to a numpy array for OpenCV processing
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image is not valid"}), 400

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    face_info = []
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Predict the face ID using the recognizer
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            name = names[id]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        face_info.append({
            'name': name,
            'confidence': confidence_text,
            'coordinates': {'x': x, 'y': y, 'w': w, 'h': h}
        })

    # Return the result as JSON
    if face_info:
        return jsonify(face_info)
    else:
        return jsonify({"message": "No faces detected"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
