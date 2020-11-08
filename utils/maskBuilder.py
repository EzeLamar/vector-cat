import sys
import cv2
import os
import numpy as np

#constants
BLACK_BG = (0, 0, 0)
WHITE_COLOR = (255,255,255)
BACKGROUND_MASK = (0,0,0)
SELECTED_MASK = (255,255,255)
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
    maskNose[:,:]= BACKGROUND_MASK
    # draw mask of noise detected
    pointsTruncated = truncateCoordinatesOfPoints(points)
    noisePoints = [[pointsTruncated[4],pointsTruncated[5],pointsTruncated[14]]]
    nosePoints = np.array([noisePoints])
    maskNose = cv2.fillPoly(maskNose, nosePoints, SELECTED_MASK)
    return maskNose
# params: frame= to get the dimensions of the frame; points= all the points of de face, including eyes
# return: a frame that contains only the MASK IN WHITE of the eyes
def getEyeMask(frame, points):
    maskEyes = np.zeros(frame.shape,np.uint8)
    maskEyes[:,:]= BACKGROUND_MASK
    # draw mask of eyes detected
    leftEyePoints = getLeftEyeDots(points)
    rightEyePoints = getRightEyeDots(points)
    ellipseLeftEye = cv2.fitEllipse(np.array(leftEyePoints))  #from 0 to 4 first eye
    cv2.ellipse(maskEyes,ellipseLeftEye,SELECTED_MASK,-1)
    ellipseRightEye = cv2.fitEllipse(np.array(rightEyePoints))  #from 10 to 13 second eye
    cv2.ellipse(maskEyes,ellipseRightEye,SELECTED_MASK,-1)
    return maskEyes

# TODO
# params: frame= to get the dimensions of the frame; points= all the points of de face, including eyes
# return: a new frame that contains only the MASK IN WHITE of the face
def getFaceMask(frame, points):
    maskFace = np.zeros(frame.shape,np.uint8)
    maskFace[:,:]= BACKGROUND_MASK
    # draw mask of eyes & nose detected
    maskEyes = getEyeMask(frame, points)
    maskNoise = getNoseMask(frame, points)
    # draw mask of face detected
    pointsTruncated = truncateCoordinatesOfPoints(points)
    topPoint_x = int((pointsTruncated[6][0]+pointsTruncated[16][0])/2)
    topPoint_y = int((pointsTruncated[6][1]+pointsTruncated[16][1])/2)-10
    topPoint = (topPoint_x, topPoint_y)
    facePoints = [[topPoint, pointsTruncated[6],pointsTruncated[7],pointsTruncated[8],pointsTruncated[9],pointsTruncated[15],pointsTruncated[19],pointsTruncated[16]]]
    facePoints = np.array([facePoints])
    maskFace = cv2.fillPoly(maskFace, facePoints, SELECTED_MASK)
    maskFace[np.where((maskEyes==SELECTED_MASK).all(axis=2))] = BACKGROUND_MASK
    maskFace[np.where((maskNoise==SELECTED_MASK).all(axis=2))] = BACKGROUND_MASK
    return maskFace
    #TODO falta obtener la mascara de la cara
    #TODO falta hacer la resta entre mascaras, para eliminar los ojos

# param: get the frame frame
# return: the frame cropped
def cropCascade(frame, face_cascade, model):
    # values to return
    points = []
    croppedFace = None
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

            #NEW
            # crop face to face detected
            gray_crop_face = gray[y:y+h, x:x+w]
            color_crop_face = frame[y:y+h, x:x+w]
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

            for i, co in enumerate(keypoints[0][0::2]):
                points.append((co, keypoints[0][1::2][i]))
            #ENDNEW

            # get the image on grayScale in 96x96 size
            croppedFace = frame[y:y+h, x:x+w]
            croppedFace = cv2.resize(croppedFace, (96, 96), interpolation = cv2.INTER_AREA)
            return croppedFace, points