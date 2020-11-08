#from model import *
import cv2
import numpy as np
import os
import sys
import time


if __name__ == "__main__":
    # constants
    PATH_CASCADE = '../../cascade/haarcascade_frontalcatface.xml'

    # default value options
    optionSelected = 'color'
    resizeImage = False

    # default eye colors
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),     #RED
        ([100, 17, 15], [200, 56, 50]),       #
        ([25, 146, 190], [62, 174, 250]),   #
        ([103, 86, 65], [145, 133, 128])    #
    ]
    boundariesHSV = [
        ([8,0,50],[20,255,255]),   #naranja HUE 0-179; Sat 0-255; Val 0-255
        ([21,0,50],[30,255,255]),   #amarillo HUE 0-179; Sat 0-255; Val 0-255
        ([31,0,50],[65,255,255]),   #GREEN HUE 0-179; Sat 0-255; Val 0-255
        ([80,0,50],[115,255,255]),   #BLUE HUE 0-179; Sat 0-255; Val 0-255
    ]

    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print('\nUsage: openImageWithCascade input_file <COLOR_OPTION> <RESIZE_IMAGE>')
        print('\tCOLOR_OPTION: (color) | grey')
        print('\tRESIZE_IMAGE: (true) | false')
        exit()

    if len(sys.argv) >= 4:
        optionSelected = sys.argv[2]
    if len(sys.argv) == 5:
        resizeImage = sys.argv[3] == 'true'

    # Face cascade to detect faces
    print('Loading Haar Cascade for detection..')
    face_cascade = cv2.CascadeClassifier(os.path.abspath(PATH_CASCADE))
    print('Loading Done.')
    # Keep looping
    print('Detecting and Cropping Face...')

    # load image file to work..
    actualImage = sys.argv[1]
    frame = cv2.imread('../../media/input/'+actualImage+'.jpg',cv2.IMREAD_COLOR)

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

        #----MODIFICACTIONS FOR TENSORFLOW_MODEL----
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
        #----END TENSORFLOW_MODEL MODIF----

        #TODO falta chequear que no se caiga en ancho y alto...

        if optionSelected == 'color':
            croppedFace = color[y:y+h, x:x+w]
        elif optionSelected == 'grey':
            croppedFace = gray[y:y+h, x:x+w]

        if resizeImage:
            croppedFace = cv2.resize(croppedFace, (96, 96), interpolation = cv2.INTER_AREA)

        # loop over the boundaries
        frameHSV = croppedFace #cv2.cvtColor(croppedFace, cv2.COLOR_BGR2HSV)
        for (lower, upper) in boundariesHSV:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(frameHSV, lower, upper)
            output = cv2.bitwise_and(frameHSV, frameHSV, mask = mask)
            # show the images
            cv2.imshow("images", np.hstack([frameHSV, output]))
            cv2.waitKey(0)

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
