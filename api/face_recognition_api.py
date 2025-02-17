import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# MongoDB Connection
MONGO_URI = "mongodb+srv://samosa:Laudalele@mine.nlznt.mongodb.net/?retryWrites=true&w=majority&appName=mine"
client = MongoClient(MONGO_URI)
db = client["face_recognition_db"]
students_collection = db["students"]

def load_known_faces():
    known_faces = []
    known_names = []

    for student in students_collection.find():
        name = student["name"]
        image_data = base64.b64decode(student["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_faces.append(encoding[0])
            known_names.append(name)

    return known_faces, known_names

@app.route("/register", methods=["POST"])
def register_face():
    data = request.json
    name = data.get("name")
    image_data = data.get("image")

    if not name or not image_data:
        return jsonify({"error": "Name and image are required"}), 400

    image_base64 = image_data.split(",")[1]

    students_collection.insert_one({"name": name, "image": image_base64})

    return jsonify({"message": "Face registered successfully"})


@app.route("/recognize", methods=["POST"])
def recognize_face():
    known_face_encodings, known_face_names = load_known_faces()

    data = request.json
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "Image is required"}), 400

    image_data = base64.b64decode(image_data.split(",")[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_encodings = face_recognition.face_encodings(rgb_img)

    if not face_encodings:
        return jsonify({"message": "No face detected"}), 400

    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    if True in matches:
        matched_index = matches.index(True)
        student_name = known_face_names[matched_index]
        return jsonify({"status": "Present", "name": student_name})
    else:
        return jsonify({"status": "Absent", "name": "Unknown"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
