# Real-time face detection

import cv2
from random import randrange

# Pre-trained data on face frontals
har_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Importing an image to detect faces
webcam = cv2.VideoCapture(0)

# Iterating over the video frames
while True:
    successful_frame_read, frame = webcam.read()

    # Converting to grayscale
    grayscale_cam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    face_coordinates = har_cascade.detectMultiScale(grayscale_cam)

    # Drawing rectangles around the faces
    for (x,y,w,h) in face_coordinates: # This loop is used so that multiple faces can be detected in an image at once
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(128,256), randrange(128,256),randrange(128,256)), 10) # Here, "(0,0,255)" is used to represent the color of the square

    # Showing the webcam footage
    cv2.imshow("Face Detector", frame)
    key = cv2.waitKey(1) # Video keeps refreshing the frames after every 1ms
    
    # Stop if Spacebar is pressed
    if key == 32:
        break

print("Code Completed!")