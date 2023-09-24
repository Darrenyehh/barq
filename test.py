import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import numpy as numpy



cap = cv2.VideoCapture(0)
handDect = HandDetector(maxHands=1)
classifier = Classifier("C:\\Users\\darrr\\OneDrive\\Desktop\\Proj\\voicebox\\Model\\keras_model.h5","C:\\Users\\darrr\\OneDrive\\Desktop\\Proj\\voicebox\\Model\\labels.txt")
offset = 20
imageSize = 300
folder = "images/3"
counter = 0
index = -1
t = 0
tempGesture = -1
labels = ["YES", "PLEASE"]

#fix the error where it crashes if the hand gets too close
while True:
    success, imga = cap.read()
    hands, img = handDect.findHands(img= imga)
    if hands:
        hand = hands[0]
        x,y,w,h = hand ['bbox']

        imgWhite = numpy.ones((imageSize,imageSize,3), numpy.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x- offset:x + w+ offset]

        imgCropShape = imgCrop.shape

        aspectRat = h/w
        
        #tweak it so horizontal size is also adjusted
        if aspectRat > 1:
            k = imageSize/h
            newWidth = int(math.ceil(k*w))
            newImgSize = cv2.resize(imgCrop, (newWidth, imageSize))
            wGap = int(math.ceil((imageSize - newWidth) / 2))
            imgWhite[:, wGap:wGap+newWidth] = newImgSize
            prediction, index = classifier.getPrediction(imgWhite)
        else:
            k = imageSize/w 
            newHeight = int(math.ceil(k*h))
            newImgSize = cv2.resize(imgCrop, (imageSize, newHeight))
            hGap = int(math.ceil((imageSize - newHeight) / 2))
            imgWhite[hGap:hGap+newHeight, :] = newImgSize
            prediction, index = classifier.getPrediction(imgWhite)
            
        
        if tempGesture == index:
            t += .25
            if t >= 2:
                print(labels[tempGesture])
                break
        else:
            tempGesture = index
            t = 0
        print(t)

        cv2.imshow("imageCrop",imgCrop)
        cv2.imshow("imageWhite", imgWhite)



    cv2.imshow("image", imga)
