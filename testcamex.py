# import cv2
#
# filename="imgmirror.jpg"
# img= cv2.imread('image.jpg')
# res= img.copy()
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         res[i][img.shape[1]-j-1]= img[i][j]
#
# cv2.imshow('image', res)
# cv2.imwrite(filename,res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
#
# img = cv2.imread("no entry.png")
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("image ori", img)
# cv2.imshow("image gray", gray)
# filename="noentrygray.jpg"
# cv2.imwrite(filename,gray)
# cv2.waitKey(0)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import cv2

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

sampleNum=0

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


# def getCalssName(classNo):
#     if classNo == 0:
#         return 'No Entry'
#     elif classNo == 1:
#         return 'Turn Right'
#     elif classNo == 2:
#         return 'Turn Left'
#     elif classNo == 3:
#         return 'Go Ahead'

# cascLeft = "all.xml"
# cascRight = "all.xml"
# cascStop = "all.xml"
cascLeft = "turnLeft_ahead.xml"
cascRight = "turnRight_ahead.xml"
cascStop = "stopsign_classifier.xml"
#speedLimit = "lbpCascade.xml"
leftCascade = cv2.CascadeClassifier(cascLeft)
rightCascade = cv2.CascadeClassifier(cascRight)
stopCascade = cv2.CascadeClassifier(cascStop)
#speedCascade = cv2.CascadeClassifier(speedLimit)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

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
    # speed = speedCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30)
    # )

    # Draw a rectangle around the faces
    for (x, y, w, h) in left:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
        prediction = model.predict(cropped_img)
        #sampleNum = sampleNum + 1
        rambu = ('Stop', 'Turn Right', 'Turn Left')
        maxindex = rambu[int(np.argmax(prediction))]

        cv2.putText(frame, maxindex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.imwrite("TrainingImage\ " + str(sampleNum) + ".jpg", frame)
        # if probabilityValue > threshold:
        #     cv2.putText(frame, str(tessss) + "%", (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in right:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
        prediction = model.predict(cropped_img)
        #sampleNum = sampleNum + 1
        rambu = ('Stop', 'Turn Right', 'Turn Left')
        maxindex = rambu[int(np.argmax(prediction))]

        cv2.putText(frame, maxindex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.imwrite("TrainingImage\ " + str(sampleNum) + ".jpg", frame)
        #probabilityValue = np.amax(prediction)
        # if probabilityValue > threshold:
        #     cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in stop:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
        prediction = model.predict(cropped_img)
        #sampleNum = sampleNum + 1
        rambu = ('Stop', 'Turn Right', 'Turn Left')
        maxindex = rambu[int(np.argmax(prediction))]

        cv2.putText(frame, maxindex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.imwrite("TrainingImage\ " + str(sampleNum) + ".jpg", frame)

    # for (x ,y, w, h) in speed:
    #     cv2.rectangle(frame, (x ,y), (x+w, y+h), (0, 255, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (32, 32)), -1), 0)
    #     prediction = model.predict(cropped_img)
    #
    #     rambu = ('Stop', 'Turn Right', 'Turn Left', 'Max Speed 50')
    #     maxindex = rambu[int(np.argmax(prediction))]
    #
    #     cv2.putText(frame, maxindex, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


