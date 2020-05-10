from model import *
import cv2
import sys
import os
import numpy as np

#constants
BLACK_BG = (0, 0, 0)
WHITE_COLOR = (255,255,255)
boundariesEyeHSV = [
    ([14,40,95],[19,211,233]),   #naranja
    ([20,0,50],[30,255,255]),   #amarillo
    ([31,0,50],[65,255,255]),   #verde
    ([80,21,50],[115,160,255]),  #celeste
]

if len(sys.argv) < 3:
    print('\nUsage: colorEyeCalibrator INPUT_COLOR_FOLDER input_image')
    print('\tINPUT_COLOR_FOLDER: naranja | amarillo | verde | celeste | test')
    print('\tinput_image: name of image to scan; example: "image1"')
    exit()

# load image to scan..
actualFolder = 'test'    #sys.argv[1]
actualImage = sys.argv[3]
picturePath = './media/input/ojos/'+actualFolder+'/'+actualImage+'.jpg'

# Load the model built in the previous step
model = load(os.path.abspath(sys.argv[1]))

# Face cascade to detect faces
face_cascade = cv2.CascadeClassifier(os.path.abspath(sys.argv[2]))

# Process current original
original = cv2.imread(picturePath,cv2.IMREAD_COLOR)
original = cv2.flip(original, 1)
originalDots = original.copy()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
maskEyes = np.zeros(original.shape,np.uint8)
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
    color_crop_face = original[y:y+h, x:x+w]
    maskEyes_crop_face = maskEyes[y:y+h, x:x+w]

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

    # Map the Keypoints back to the original image
    face_resized_color = cv2.resize(color_crop_face, (96, 96), interpolation = cv2.INTER_AREA)
    face_resized_maskEyes = cv2.resize(maskEyes_crop_face, (96, 96), interpolation = cv2.INTER_AREA)

    # Pair them together
    for i, co in enumerate(keypoints[0][0::2]):
        points.append((co, keypoints[0][1::2][i]))

    # Add KEYPOINTS to the original
    for keypoint in points[0:21]:
        cv2.circle(face_resized_color, keypoint, 1, (0,255,0), 1)
    originalDots[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
# show original picture with dots
cv2.imshow("face detected", originalDots)
# draw mask of eyes detected
leftEyePoints = [points[0], points[1], points[2], points[3], points[0]]
rightEyePoints = [points[10], points[11], points[12], points[13], points[10]]
ellipseLeftEye = cv2.fitEllipse(np.array(leftEyePoints))  #from 0 to 4 first eye
cv2.ellipse(face_resized_maskEyes,ellipseLeftEye,WHITE_COLOR,-1)
ellipseRightEye = cv2.fitEllipse(np.array(rightEyePoints))  #from 10 to 13 second eye
cv2.ellipse(face_resized_maskEyes,ellipseRightEye,WHITE_COLOR,-1)
maskEyes[y:y+h, x:x+w] = cv2.resize(face_resized_maskEyes, original_shape, interpolation = cv2.INTER_CUBIC)
cv2.imshow('mascara ojos', maskEyes)
# use maskEyes to filter only the eyes of original image
originalEyes = original.copy()
originalEyes[np.where((maskEyes!=WHITE_COLOR).all(axis=2))] = BLACK_BG


# initial values
scanningColors = True
indexColor = 0  #starts with naranja
totalColorArea = 0
colorArea = [0,0,0,0]   #area of naranja, amarillo, verde, celeste
while scanningColors:
    hsv = cv2.cvtColor(originalEyes, cv2.COLOR_BGR2HSV)

    (lower, upper) = boundariesEyeHSV[indexColor]
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
        colorArea[indexColor] += cv2.contourArea(c)
        #cv2.drawContours(originalEyes,[c], 0, (0,0,0), 2)
    totalColorArea += colorArea[indexColor]
    cv2.imshow("maskED", maskED)
    if indexColor == 0:
        etiqueta = 'naranja'
    elif indexColor == 1:
        etiqueta = 'amarillo'
    elif indexColor == 2:
        etiqueta = 'verde'
    elif indexColor == 3:
        etiqueta = 'celeste'
    cv2.imshow(etiqueta, coloured)

    key = cv2.waitKey(0)
    if key == 27:
        break
    indexColor += 1
    if indexColor > 3:
        scanningColors = False

# detect color of the eyes
biggestEyeColor = colorArea[0]
result = 'naranja'
if biggestEyeColor < colorArea[1]:
    biggestEyeColor = colorArea[1]
    result = 'amarillo'
if biggestEyeColor < colorArea[2]:
    biggestEyeColor = colorArea[2]
    result = 'verde'
if biggestEyeColor < colorArea[3]:
    biggestEyeColor = colorArea[3]
    result = 'celeste'

print('naranja: '+str((colorArea[0]*100)/totalColorArea)+' %')
print('amarillo: '+str((colorArea[1]*100)/totalColorArea)+' %')
print('verde: '+str((colorArea[2]*100)/totalColorArea)+' %')
print('celeste: '+str((colorArea[3]*100)/totalColorArea)+' %')
print('\nEl color que predomina es: '+result)
cv2.destroyAllWindows()