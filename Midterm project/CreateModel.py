# ALireza Mousavi 983613053
# Import the necessary modules
import cv2
import numpy as np
from skimage import color, transform, feature
from sklearn import svm, metrics, model_selection
import joblib
import glob

path = "genki4k/files/*.jpg"
# Get a list of image file names
filenames = glob.glob(path)
# Load the face images and their labels from the dataset
images = []
for file in filenames:
    # Read the image
    img = cv2.imread(file)
    # Append the image to the list
    images.append(img)

labels = np.loadtxt("genki4k/labels.txt")
y_target = labels[:,0]

# Initialize an empty list to store the feature vectors
features = []

# Load a pre-trained face detector model
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loop over the face images and extract HOG and LBP features
for image in images:
    # Detect faces in image
    flag = False
    faces = face_detector.detectMultiScale(image)
    for (x,y,w,h) in faces:
        # Crop face from image
        face = image[y:y+h,x:x+w]
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
        features.append(feature_vector)
        flag = True
        break
    #####
    if not flag:
        # Convert the image to grayscale
        gray = color.rgb2gray(image)
        # Resize the image to a fixed size
        resized = transform.resize(gray, (64, 64))
        # Extract HOG features
        hog_features = feature.hog(resized, orientations=20, pixels_per_cell=(8,8), cells_per_block=(2, 2))
        # Extract LBP features
        lbp_features = feature.local_binary_pattern(resized, P=8, R=1)
        # Compute the histogram of LBP features
        lbp_hist = np.histogram(lbp_features.ravel(), bins=256)[0]
        # Concatenate HOG and LBP features into a single feature vector
        feature_vector = np.hstack((hog_features, lbp_hist))
        # Append the feature vector to the list
        features.append(feature_vector)

# Convert the list of feature vectors to a numpy array
features = np.array(features)
print(features.shape)


# Split the feature vectors and labels into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features,y_target,
    test_size=0.2,
    random_state=42)
    #stratify=labels)

# Create a SVM classifier
clf = svm.SVC(C=1.0,kernel="poly")
# clf = svm.SVC(C=1.0,kernel="rbf",gamma="scale")

# Train the SVM classifier using the training feature vectors and labels
clf.fit(X_train,y_train)

# Evaluate the performance of the SVM classifier on testing feature vectors and labels
y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test,y_pred)

precision = metrics.precision_score(y_test,y_pred,average="macro")

recall = metrics.recall_score(y_test,y_pred,average="macro")

f1_score = metrics.f1_score(y_test,y_pred,average="macro")

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

# Save the trained SVM model
joblib.dump(clf,"svm_model.pkl")