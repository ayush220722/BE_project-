
import numpy as np
from plyer import notification
import datetime
import cv2
import face_recognition
import threading

def notify_me(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=10
    )

fire_cascade = cv2.CascadeClassifier("D:/BE PROJECT/face recognization/test/fire_detection.xml")

known_image1 = face_recognition.load_image_file("D:/BE PROJECT/face recognization/ayush.jpg")
known_face_encoding1 = face_recognition.face_encodings(known_image1)[0]
known_image2 = face_recognition.load_image_file("D:/BE PROJECT/face recognization/rakesh.jpg")
known_face_encoding2 = face_recognition.face_encodings(known_image2)[0]
known_image3 = face_recognition.load_image_file("D:/BE PROJECT/face recognization/AG.jpg")
known_face_encoding3 = face_recognition.face_encodings(known_image3)[0]

known_face_encodings = [known_face_encoding1, known_face_encoding2, known_face_encoding3]
known_face_names = ["Ayush Naik", "Rakesh Mali", "Atharv Gawande"]

def face_rec(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            curr_time = datetime.datetime.now()
            curr_time = str(curr_time)
            known_encoding = np.array(known_face_encodings[first_match_index])
            unknown_encoding = np.array(face_encoding)
            face_dis = np.linalg.norm(known_encoding - unknown_encoding)
            notify_me(name + " SPOTTED", "CURRENT TIME: " + curr_time)  # notification

            for (top, right, bottom, left), name in zip(face_locations, [name]):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

def fire(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fires = fire_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        curr_time = datetime.datetime.now()
        curr_time = str(curr_time)
        notify_me("FIRE SPOTTED", "CURRENT TIME: " + curr_time)

def main():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        
        face_thread = threading.Thread(target=face_rec, args=(frame.copy(),))
        fire_thread = threading.Thread(target=fire, args=(frame.copy(),))

        
        face_thread.start()
        fire_thread.start()

       
        face_thread.join()
        fire_thread.join()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

