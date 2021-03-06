#from model import *
import cv2
import numpy as np
import os
import sys
import time


if __name__ == "__main__":
    # default value options
    optionSelected = 'color'
    resizeImage = True

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('\nUsage: openImageWithCascade <haar_cascade_classifier> input_file <COLOR_OPTION> <RESIZE_IMAGE>')
        print('\tCOLOR_OPTION: (color) | grey')
        print('\tRESIZE_IMAGE: (true) | false')
        exit()

    if len(sys.argv) >= 4:
        optionSelected = sys.argv[3]
    if len(sys.argv) == 5:
        resizeImage = sys.argv[4] == 'true'

    # Face cascade to detect faces
    print('Loading Haar Cascade for detection..')
    face_cascade = cv2.CascadeClassifier(os.path.abspath(sys.argv[1]))
    print('Loading Done.')
    # Keep looping
    print('Detecting and Cropping Face...')

    # load image file to work..
    actualImage = sys.argv[2]
    frame = cv2.imread('./media/input/'+actualImage,cv2.IMREAD_COLOR)

    # Process current frame
    color = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    if len(faces) == 0:
        print('faces not detected. bye bye')
        exit()
    for (x, y, w, h) in faces:
        # Grab the face
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_face.shape # A Copy for future reference
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # draw a blue rectangle on the image
        x -= 10
        y -= 10
        w += 20
        h += 20
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        #TODO falta chequear que no se caiga en ancho y alto...

        if optionSelected == 'color':
            croppedFace = color[y:y+h, x:x+w]
        elif optionSelected == 'grey':
            croppedFace = gray[y:y+h, x:x+w]

        if resizeImage:
            croppedFace = cv2.resize(croppedFace, (96, 96), interpolation = cv2.INTER_AREA)
        cv2.imshow("cropped", croppedFace)
        #write new cropped_image
        path = './media/output/'
        cv2.imwrite(os.path.join(path, 'image{}.png'.format(actualImage.split('.')[0])), croppedFace)
        isCropping = False

        # Display the section marked to crop over image
        #frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.imshow("Facial Keypoints", frame)
    #Close any open windows
    print('Cropping Done.')
    print('Bye bye')
    cv2.destroyAllWindows()
