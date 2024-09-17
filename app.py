from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from face_detection import detect_faces  # Import your face detection function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_faces', methods=['POST'])
def detect_faces_route():
    file = request.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = detect_faces(img)  # Replace this with your actual face detection function

    return jsonify({'faces': faces})

if __name__ == '__main__':
    app.run(debug=True)
