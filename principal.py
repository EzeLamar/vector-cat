import cv2
import sys
import os
import numpy as np
from utils.modelNN.model import load
from utils.faceSpotDetection.faceSpotsDetector import generateAllColoursMask
from utils.eyesDetection.colorEyeDetector import getPrincipalEyeColourFromFrame

# constants
PATH_MODAL = './model/prueba100fotos_inv_2.sh'
PATH_CASCADE = './cascade/haarcascade_frontalcatface.xml'

# Load the model built in the previous step
model = load(os.path.abspath(PATH_MODAL))

# Face cascade to detect faces
face_cascade = cv2.CascadeClassifier(os.path.abspath(PATH_CASCADE))

frame = cv2.imread('./media/input/spots/negro/cat1.jpg',cv2.IMREAD_COLOR)
cv2.imshow('originalFrame', frame)
cv2.waitKey(0)
maskFrame = generateAllColoursMask(frame)
getPrincipalEyeColourFromFrame(frame, model, face_cascade)
cv2.imshow('faceMask', maskFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()
