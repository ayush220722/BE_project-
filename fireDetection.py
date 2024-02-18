# import cv2
# from playsound import playsound


# fire_cascade = cv2.CascadeClassifier("C:\\Program Files\\Python37\\Lib\\site-packages\\cv2\\data\\fire_detection.xml")

# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

#     for (x,y,w,h) in fire:
#         cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         print("fire is detected")
#         playsound('audio.mp3')

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import cv2
from playsound import playsound

# Load the cascade classifier for fire detection
fire_cascade = cv2.CascadeClassifier("D:\BE PROJECT\fire detection-20240208T133618Z-001\fire detection-20240208T133618Z-001\fire detection\fire_detection.xml")

# Open the default camera (usually webcam)
cap = cv2.VideoCapture(0)

# Loop to continuously capture frames from the camera
while True:
    # Read the frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect fires in the frame
    fires = fire_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through the detected fires and draw rectangles around them
    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("Fire detected!")  # Optional: Print a message when fire is detected
        # Play the audio file when fire is detected
        playsound("D:\BE PROJECT\fire detection-20240208T133618Z-001\fire detection-20240208T133618Z-001\fire detection\audio.mp3")
    
    # Display the frame
    cv2.imshow('Fire Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()




    

