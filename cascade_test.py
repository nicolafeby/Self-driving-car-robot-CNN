import cv2

cam = cv2.VideoCapture(0)
harcascadePath = "haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier(harcascadePath)
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #incrementing sample number
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder TrainingImage
        cv2.imwrite("TrainingImage\ "+ str(sampleNum) + ".jpg",img)
        #display the frame
        cv2.imshow('Ambil Gambar',img)
        #wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 60
    elif sampleNum>60:
        break
cam.release()
cv2.destroyAllWindows()