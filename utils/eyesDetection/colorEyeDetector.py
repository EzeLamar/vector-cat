import sys
sys.path.insert(1, '../modelNN')
from model import *
import cv2
import os
import numpy as np
from utils.maskGenerator import getEyeMask
from utils.maskGenerator import getNoseMask
from utils.maskGenerator import getFaceMask

#constants
BLACK_BG = (0, 0, 0)
WHITE_COLOR = (255,255,255)
eyeColours = [
    'naranja',
    'amarillo',
    'verde',
    'celeste',
]
boundariesEyeHSV = {
    'naranja' : ([14,40,95],[19,211,233]),   #naranja
    'amarillo' : ([20,0,50],[30,255,255]),   #amarillo
    'verde' : ([31,0,50],[65,255,255]),   #verde
    'celeste' : ([80,21,50],[115,160,255]),  #celeste
}

def detectEyesFromFrame(originalFrame, model, face_cascade):
    # percentage of colours detected
    colorArea = {
        'naranja': 0,
        'amarillo': 0,
        'verde': 0,
        'celeste': 0,
    }
    # percentage of colours to return
    percentageColorArea = {
        'naranja': 0,
        'amarillo': 0,
        'verde': 0,
        'celeste': 0,
    }

    # Process current originalFrame
    originalDots = originalFrame.copy()
    gray = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)
    maskEyes = np.zeros(originalFrame.shape,np.uint8)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
    points = []
    if len(faces) == 0:
        print('\tCat faces not detected')
        exit()
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

        # crop face to face detected
        gray_crop_face = gray[y:y+h, x:x+w]
        color_crop_face = originalFrame[y:y+h, x:x+w]
        maskEyes_crop_face = np.zeros(color_crop_face.shape,np.uint8)

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_crop_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_crop_face.shape # A Copy for future reference
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # Predicting the keypoints using the model
        keypoints = model.predict(face_resized)
        # De-Normalize the keypoints values
        keypoints = keypoints * 48 + 48

        # Map the Keypoints back to the originalFrame image
        face_resized_color = cv2.resize(color_crop_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_maskEyes = cv2.resize(maskEyes_crop_face, (96, 96), interpolation = cv2.INTER_AREA)
        # Pair them together
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        # Add KEYPOINTS to the originalFrame
        for keypoint in points[0:21]:
            cv2.circle(face_resized_color, keypoint, 1, (0,0,255), 1)
        originalDots[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
    # show originalFrame picture with dots
    cv2.imshow("face detected", originalDots)
    # draw mask of eyes detected
    frameNose = getNoseMask(face_resized_maskEyes, points)
    cv2.imshow("noseMask", frameNose)
    frameFace = getFaceMask(face_resized_maskEyes, points)
    cv2.imshow("faceMask", frameFace)
    face_resized_maskEyes = getEyeMask(face_resized_maskEyes, points)
    cv2.imshow("face_resized_maskEyes", face_resized_maskEyes)

    maskEyes[y:y+h, x:x+w] = cv2.resize(face_resized_maskEyes, original_shape, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('mascara ojos', maskEyes)
    # use maskEyes to filter only the eyes of originalFrame image
    originalEyes = originalFrame.copy()
    originalEyes[np.where((maskEyes!=WHITE_COLOR).all(axis=2))] = BLACK_BG


    # initial values
    totalColorArea = 0
    for actualColour in eyeColours:
        hsv = cv2.cvtColor(originalEyes, cv2.COLOR_BGR2HSV)

        (lower, upper) = boundariesEyeHSV[actualColour]
        lower_blue = np.array([lower[0], lower[1], lower[2]])
        upper_blue = np.array([upper[0], upper[1], upper[2]])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        result = cv2.bitwise_and(originalEyes, originalEyes, mask=mask)
        coloured = originalEyes.copy()
        coloured[mask == 0] = BLACK_BG  #black background
        # apply erode & dilate to mask. After detect contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        maskED = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=0)
        cnts = cv2.findContours(maskED, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # calculate area of this color in the image
        for c in cnts:
            colorArea[actualColour] += cv2.contourArea(c)
            #cv2.drawContours(originalEyes,[c], 0, (0,0,0), 2)
        totalColorArea += colorArea[actualColour]
        #cv2.imshow("maskED", maskED)
        #cv2.imshow(actualColour, coloured)

        """ key = cv2.waitKey(0)
        if key == 27:
            break """

    for colour in eyeColours:
        if totalColorArea != 0:
            percentageColorArea[colour] = colorArea[colour]*100/totalColorArea
    return percentageColorArea

def getPrincipalEyeColourFromFrame(originalFrame, model, face_cascade):
    percentageColoursDetected = detectEyesFromFrame(originalFrame, model, face_cascade)

    # show percentage of every colour & calcultate the principal colour
    principalEyeColor = None
    print('\nPorcentajes detectados de los colores buscados:')
    for colour in eyeColours:
        print('\t'+colour+': '+str(percentageColoursDetected[colour])+' %')
        if principalEyeColor == None:
            principalEyeColor = colour
        elif percentageColoursDetected[principalEyeColor] < percentageColoursDetected[colour]:
            principalEyeColor = colour
    print('-------------------------------')
    print('El color de ojos que mas aparece es "'+principalEyeColor+'"')
    print('-------------------------------')
