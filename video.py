import cv2
import pafy
import numpy as np

path_classifier = './premodels/haarcascade_frontalface_default.xml'


## Get user option
print('Input video URL to analyse or 0 to use webcam')
inp = input()
if inp=='0':
    inp=int(inp)
else:
    vPafy = pafy.new(inp)
    play = vPafy.getbest(preftype='mp4')
    inp = play.url

cap = cv2.VideoCapture(inp)
cap.set(3, 480) #set width of the frame
cap.set(4, 640) #set height of the frame

## Fixed Values
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX

## LOAD CAFFE MODELS
age_net = cv2.dnn.readNetFromCaffe('./premodels/deploy_age.prototxt', './premodels/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('./premodels/deploy_gender.prototxt', './premodels/gender_net.caffemodel')



while True:
    _, img = cap.read()
    
    face_cascade = cv2.CascadeClassifier(path_classifier)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 7)
    

    if(len(faces)>0):
    # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Get Face
            face_img = img[y:y+h, h:h+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227,227), MODEL_MEAN_VALUES, swapRB=False)

    
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
    
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]


            ## Display age and gender
            overlay_text = "%s %s" % (gender,age)
            cv2.putText(img, overlay_text, (x,y),font,1,(255,255,255), 2, cv2.LINE_AA)



    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()