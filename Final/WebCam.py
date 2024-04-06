# ALireza Mousavi 983613053
# Import the necessary modules
import cv2
import numpy as np
import CreateModel


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 5
color = (0,255,255)
thickness = 10


model = CreateModel.train()
cap = cv2.VideoCapture(1)
while True:
    ret,frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(gray , 1.1 , 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
            f = frame[y:y+h , x:x+w]
            v = cv2.resize(f, (32,32), interpolation = cv2.INTER_AREA)
            data = []
            data.append(v)
            d = np.stack(data, axis=0) / 255.0
            p = model.predict(d)
            text = 'laugh'
            if p[0][0] > p[0][1]:
                text = 'careless'
            # frame = cv2.putText(frame, text, (x,y), font, fontScale, color, thickness, cv2.Line_AA)
            frame = cv2.putText(frame, text, (x, y+h+40), font, fontScale, color, thickness)


        # Display frame with annotations
        cv2.imshow("Face Detection and Laughing Detection",frame)
        key = cv2.waitKey(1) & 0xFF

        # Press q to quit
        if key == ord("q"):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()