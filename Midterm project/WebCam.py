# ALireza Mousavi 983613053
# Import the necessary modules
import cv2
import numpy as np
from skimage import color, transform, feature
from sklearn import svm
import joblib

# Connect mobile camera to computer via cable
# Open DroidCam App on both mobile and pc and connect them together
# and capture frames in loop
cap = cv2.VideoCapture(1) # Change index according to your device
# Open webcam and capture frames in a loop
while True:
    # Read a frame from webcam
    ret,frame = cap.read()
    if ret:

        # Load a pre-trained face detector model
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Detect faces in frame
        faces = face_detector.detectMultiScale(frame)

        # Loop over bounding boxes of faces
        for (x,y,w,h) in faces:
            # Crop face from frame
            face = frame[y:y+h,x:x+w]
            # Convert face to grayscale
            gray = color.rgb2gray(face)
            # Resize face to fixed size
            resized = transform.resize(gray,(64,64))
            # Extract HOG features from face
            hog_features = feature.hog(resized,orientations=20,pixels_per_cell=(8,8),cells_per_block=(2,2))
            # Extract LBP features from face
            lbp_features = feature.local_binary_pattern(resized,P=8,R=1)
            # Compute histogram of LBP features
            lbp_hist = np.histogram(lbp_features.ravel(),bins=256)[0]
            # Concatenate HOG and LBP features into single feature vector
            feature_vector = np.hstack((hog_features,lbp_hist))

            # Load trained SVM model
            svm_model = joblib.load("svm_model.pkl")

            # Predict expression of face
            y_pred = int(svm_model.predict([feature_vector])[0])

            # Draw bounding box and label on frame
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.putText(frame,expression+"",(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            if y_pred == 1:
                cv2.putText(frame, 'SMILING', (x, y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 255, 0))
            else :
                cv2.putText(frame, 'NOT SMILING', (x, y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 255))
        # Display frame with annotations
        cv2.imshow("Face Detection and Laughing Detection",frame)
        key = cv2.waitKey(1) & 0xFF

        # Press q to quit
        if key == ord("q"):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()