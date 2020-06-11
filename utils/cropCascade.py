import sys
import cv2
import os
import numpy as np

#constants
BLACK_BG = (0, 0, 0)
WHITE_COLOR = (255,255,255)
"""     'inEye_L',0
    'outEye_L',1
    'upEye_L',2
    'botEye_L',3
    'holeNose_L',4
    'botNose',5
    'outEar:L',6
    'upCheek_L',7
    'botCheek_L',8
    'botMouth',9
    'inEye_R',10
    'outEye_R',11
    'upEye_R',12
    'botEye_R',13
    'holeNose_R',14
    'botCheek_R',15
    'outEar_R',16
    'upNose_L',17
    'upNose_R',18
    'upCheek_R',19
    'centerMouth',20 
"""

def truncateCoordinatesOfPoints(points):
    pointsTruncated = []
    for (x,y) in points:
        pointsTruncated.append((int(x),int(y)))
    return pointsTruncated

def getLeftEyeDots(points):
    return [points[0], points[1], points[2], points[3], points[0]]

def getRightEyeDots(points):
    return [points[10], points[11], points[12], points[13], points[10]]
#TODO
def getNoseMask(frame, points):
    maskNose = np.zeros(frame.shape,np.uint8)
    # draw mask of noise detected
    pointsTruncated = truncateCoordinatesOfPoints(points)
    noisePoints = [[pointsTruncated[4],pointsTruncated[5],pointsTruncated[14]]]
    nosePoints = np.array([noisePoints])
    maskNose = cv2.fillPoly(maskNose, nosePoints, WHITE_COLOR)
    return maskNose
# params: frame= to get the dimensions of the frame; points= all the points of de face, including eyes
# return: a frame that contains only the MASK IN WHITE of the eyes
def getEyeMask(frame, points):
    maskEyes = np.zeros(frame.shape,np.uint8)
    # draw mask of eyes detected
    leftEyePoints = getLeftEyeDots(points)
    rightEyePoints = getRightEyeDots(points)
    ellipseLeftEye = cv2.fitEllipse(np.array(leftEyePoints))  #from 0 to 4 first eye
    cv2.ellipse(maskEyes,ellipseLeftEye,WHITE_COLOR,-1)
    ellipseRightEye = cv2.fitEllipse(np.array(rightEyePoints))  #from 10 to 13 second eye
    cv2.ellipse(maskEyes,ellipseRightEye,WHITE_COLOR,-1)
    return maskEyes

# TODO
# params: frame= to get the dimensions of the frame; points= all the points of de face, including eyes
# return: a new frame that contains only the MASK IN WHITE of the face
def getFaceMask(frame, points):
    maskFace = np.zeros(frame.shape,np.uint8)
    # draw mask of eyes & nose detected
    maskEyes = getEyeMask(frame, points)
    maskNoise = getNoseMask(frame, points)
    # draw mask of face detected
    pointsTruncated = truncateCoordinatesOfPoints(points)
    facePoints = [[pointsTruncated[6],pointsTruncated[7],pointsTruncated[8],pointsTruncated[9],pointsTruncated[15],pointsTruncated[19],pointsTruncated[16]]]
    facePoints = np.array([facePoints])
    maskFace = cv2.fillPoly(maskFace, facePoints, WHITE_COLOR)
    maskFace[np.where((maskEyes==WHITE_COLOR).all(axis=2))] = BLACK_BG
    maskFace[np.where((maskNoise==WHITE_COLOR).all(axis=2))] = BLACK_BG
    return maskFace
    #TODO falta obtener la mascara de la cara
    #TODO falta hacer la resta entre mascaras, para eliminar los ojos

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