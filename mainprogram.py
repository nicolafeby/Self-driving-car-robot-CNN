from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import cv2

from testultrasonic import get_range
from motor import *

x=input()
if x == 'r':
    jalanMaju()

#############################################

#frameWidth = 640  # CAMERA RESOLUTION
#frameHeight = 480
#brightness = 180
#threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
#cap.set(3, frameWidth)
#cap.set(4, frameHeight)
#cap.set(10, brightness)


imageDimesions = (32, 32, 3)
noOfClasses = 3

no_Of_Filters = 60
size_of_Filter = (5, 5)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
# THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
size_of_Filter2 = (3, 3)
size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
no_Of_Nodes = 500  # NO. OF NODES IN HIDDEN LAYERS

model = Sequential()
model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
model.add(MaxPooling2D(pool_size=size_of_pool))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(no_Of_Nodes, activation='relu'))
model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('91model.h5')


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'No Entry'
    elif classNo == 1:
        return 'Turn Right'
    elif classNo == 2:
        return 'Turn Left'
    elif classNo == 3:
        return 'Go Ahead'

cascLeft = "turnLeft_ahead.xml"
cascRight = "turnRight_ahead.xml"
cascStop = "stopsign_classifier.xml"
leftCascade = cv2.CascadeClassifier(cascLeft)
rightCascade = cv2.CascadeClassifier(cascRight)
stopCascade = cv2.CascadeClassifier(cascStop)

while True:
    # READ IMAGE

    success, frame = cap.read()
    #frame = cv2.imread(path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    left = leftCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    right = rightCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    stop = stopCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    jarak = get_range()
    print("jarak = %.2f CM" %jarak)
    time.sleep(0.1)
    
    for (x, y, w, h) in stop:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
        prediction = model.predict(cropped_img)

        rambu = ('Stop', 'Turn Right', 'Turn Left')
        maxindex = rambu[int(np.argmax(prediction))]

        cv2.putText(frame, maxindex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if jarak <=80:
            berhenti()
        
    for (x, y, w, h) in left:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
        prediction = model.predict(cropped_img)

        rambu = ('Stop', 'Turn Right', 'Turn Left')
        maxindex = rambu[int(np.argmax(prediction))]

        cv2.putText(frame, maxindex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if jarak <=80:
            belokKiri()
        
    for (x, y, w, h) in right:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
        prediction = model.predict(cropped_img)

        rambu = ('Stop', 'Turn Right', 'Turn Left')
        maxindex = rambu[int(np.argmax(prediction))]

        cv2.putText(frame, maxindex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if jarak <=80:
            belokKanan()
        
        
    
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
