import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/Nobel/Desktop/facereg/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Nobel/Desktop/facereg/haarcascade_eye.xml')

cap = cv2.VideoCapture(0) # Capture frames to cap variable

while(True):
    # Capture frame by frame for video output
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
    face_cords = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # Detect face and read face coordinates

    # Print coordinates and draw box around face
    for(x,y,w,h) in face_cords:
        print(x,y,w,h) # Print the coordinates

        roi_gray = gray[y:y+h, x:x+w] # ycord_start, ycord_end
        gray_resize = cv2.resize(roi_gray, (100, 100)) # Resize to 100x100
        img_item = "my_face.png" # Img file name
        cv2.imwrite(img_item, gray_resize) # Save the gray image

        # Frame details for face
        color = (0, 255, 0) # Green frame color
        stroke = 2 # Line thickness
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke) # Draw rectangle frame

    # Display the video
    cv2.imshow('Face Regconition', frame) 

    # Press q to stop
    if cv2.waitKey(20) == ord('q'):
        break


# End and clear
cap.release()
cv2.destroyAllWindows()