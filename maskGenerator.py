import cv2
import sys
import os
import numpy as np
from utils.modelNN.model import load
from utils.faceSpotDetection.faceSpotsDetector import generateAllColoursMask
from utils.eyesDetection.colorEyeDetector import getPrincipalEyeColourFromFrame
from utils.maskGenerator import cropCascade, getFaceMask

# constants
PATH_MODAL = './model/prueba100fotos_inv_2.sh'
PATH_CASCADE = './cascade/haarcascade_frontalcatface.xml'
PATH_OUTPUT_MASKFACE = './media/output/'

# get the image from the params & show the original
imageName = sys.argv[1]
frame = cv2.imread('./media/input/'+imageName+'.jpg',cv2.IMREAD_COLOR)
if frame is None:
    print("Error: image not found..")
else:
    # Load the model built in the previous step
    model = load(os.path.abspath(PATH_MODAL))

    # Face cascade to detect faces
    face_cascade = cv2.CascadeClassifier(os.path.abspath(PATH_CASCADE))
    
    #show original frame before start 
    cv2.imshow('originalFrame', frame)
    cv2.waitKey(0)

    # crop the cat face (if there is a cat face) & get the mask
    cropedFrame, points = cropCascade(frame,face_cascade, model)
    maskFrame = generateAllColoursMask(cropedFrame,points)
    cv2.imshow('faceMaskColor', maskFrame)
    cv2.imwrite(os.path.join(PATH_OUTPUT_MASKFACE, '{}.png'.format(imageName)), maskFrame)
    # get the principal colour of the eyes cat
    getPrincipalEyeColourFromFrame(frame,model,face_cascade)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
