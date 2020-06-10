import sys
import cv2
import os
import numpy as np

# param: get the frame frame
# return: the frame cropped
def cropCascade(frame, face_cascade):
    # Process current frame
    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
    if len(faces) == 0:
        print('\tCat faces not detected')
        exit()
    else:
        for (x, y, w, h) in faces:
            # seting offset to rectangle that round the cat face
            x -= 10
            y -= 10
            w += 20
            h += 20
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            # get the image on grayScale in 96x96 size
            croppedFace = frame[y:y+h, x:x+w]
            croppedFace = cv2.resize(croppedFace, (96, 96), interpolation = cv2.INTER_AREA)
            return croppedFace