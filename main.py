import face_recognition
import numpy as np
import cv2
import csv
from datetime import datetime

# Initialize video capture from the first webcam
video_capture = cv2.VideoCapture(0)

# Loading known faces
krrish_image = face_recognition.load_image_file("faces/krrish.jpg")
krrish_encoding = face_recognition.face_encodings(krrish_image)[0]

virat_image = face_recognition.load_image_file("faces/virat.jpg")
virat_encoding = face_recognition.face_encodings(virat_image)[0]

rohit_image = face_recognition.load_image_file("faces/rohit.jpg")
rohit_encoding = face_recognition.face_encodings(rohit_image)[0]

sky_image = face_recognition.load_image_file("faces/sky.jpg")
sky_encoding = face_recognition.face_encodings(sky_image)[0]

known_face_encodings = [krrish_encoding, virat_encoding, rohit_encoding, sky_encoding]
known_face_names = ["Krrish", "Virat", "Rohit", "SuryaKumar"]

# List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Getting the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open a CSV file for writing attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Recognizing faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_COMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                linetype = 2
                cv2.putText(frame, name + " is Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])
    
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
