
from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv

app = Flask(__name__)

known_face_encodings = []
known_face_names = []

def load_known_faces():
    for filename in os.listdir('known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = face_recognition.load_image_file(f'known_faces/{filename}')
            encoding = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(filename.split('.')[0])

load_known_faces()

def mark_attendance(name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with open('attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    unknown_encoding = face_recognition.face_encodings(img)
    if unknown_encoding:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding[0])
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding[0])
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)
            return jsonify({'status': 'success', 'name': name})
    return jsonify({'status': 'fail', 'name': 'Unknown'})

if __name__ == '__main__':
    app.run(debug=True)
