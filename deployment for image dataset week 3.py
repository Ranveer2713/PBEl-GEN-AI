import cv2
import os
import pickle
import numpy as np

# ----------------------------
# Load Models
# ----------------------------
sc = pickle.load(open("scaling_model.pkl", "rb"))
pca = pickle.load(open("pca_model.pkl", "rb"))
model = pickle.load(open("face_recog_model.pkl", "rb"))

# ----------------------------
# Load Labels
# ----------------------------
label_dict = {}

people = [person for person in os.listdir("User_Data1")]

for i, person in enumerate(people):
    label_dict[i] = person

# ----------------------------
# Face Detector
# ----------------------------
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if detector.empty():
    print("Cannot load Haar Cascade")
    exit()

# ----------------------------
# Preprocessing Functions
# ----------------------------
def detect_face(gray):
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )
    return faces


def preprocess_face(gray, coords):
    faces = []

    for (x, y, w, h) in coords:
        face = gray[y:y+h, x:x+w]

        face = cv2.equalizeHist(face)

        face = cv2.resize(
            face,
            (80, 100)
        )

        faces.append(face)

    return faces


# ----------------------------
# Webcam Recognition
# ----------------------------
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera not found")
    exit()

cv2.namedWindow(
    "Face Recognition System",
    cv2.WINDOW_NORMAL
)

while True:

    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    face_coords = detect_face(gray)

    if len(face_coords) > 0:

        faces = preprocess_face(
            gray,
            face_coords
        )

        for i, face in enumerate(faces):

            test = face.reshape(1, -1)

            test = sc.transform(test)

            test = pca.transform(test)

            pred = model.predict(test)

            name = label_dict[pred[0]]

            x, y, w, h = face_coords[i]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Face Recognition System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()