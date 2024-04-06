import cv2
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import random
from tensorflow.keras import models, layers, datasets

main_image_directory = 'files'
grayface_image_directory = 'grayface'
resize_image_directory = 'resize'
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def create_cropped_gray_face():
    files = Path(main_image_directory).glob('*')
    for file in files:
        img = cv2.imread(str(file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray , 1.1 , 4)
        for (x,y,w,h) in faces:
            stord = img[y:y + h , x: x + w]
            cv2.imwrite(str(file).replace(main_image_directory , grayface_image_directory), stord)

def resize_image(x,y):
    files = Path(grayface_image_directory).glob('*')
    for file in files:
        img = cv2.imread(str(file))
        new_image = cv2.resize(img, (x,y), interpolation = cv2.INTER_AREA)
        cv2.imwrite(str(file).replace(grayface_image_directory , resize_image_directory), new_image)

def get_data():
    target = []
    file1 = open('labels.txt' , 'r')
    Lines = file1.readlines()
    for line in Lines:
        target.append(line.strip()[0])

    a1 = []
    t1 = []
    a2 = []
    t2 = []
    files = Path(resize_image_directory).glob('*')
    for file in files:
        if str(file)[11:15] =='nb_c':
            continue
        num = int(str(file)[11:15])
        img = cv2.imread(str(file))
        if random.randint(0,5) == 0:
            a1.append (img)
            t1.append([target[num - 1]])
        else:
            a2.append (img)
            t2.append([target[num - 1]])
    train_images = np.stack (a2 , axis=0)
    train_labels = np.stack(t2, axis=0).astype('uint8')
    test_images = np.stack (a1 , axis=0)
    test_labels = np.stack(t1, axis=0).astype('uint8')
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(train_images.shape)
    print(test_images.shape)
    return train_images, train_labels, test_images, test_labels

def train():
    # create_cropped_gray_face()
    # resize_image(32,32)
    train_images, train_labels, test_images, test_labels = get_data()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.5 , 1)
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)
    return model