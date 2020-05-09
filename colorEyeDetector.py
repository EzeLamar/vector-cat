import cv2
import sys
import os
import numpy as np

#constants
BLACK_BG = (0, 0, 0)
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
actualFolder = sys.argv[1]
actualImage = sys.argv[2]
picturePath = './media/input/ojos/'+actualFolder+'/'+actualImage+'.jpg'

scanningColors = True
indexColor = 0  #starts with naranja
totalColorArea = 0
colorArea = [0,0,0,0]   #area of naranja, amarillo, verde, celeste
while scanningColors:
    frame = cv2.imread(picturePath,cv2.IMREAD_COLOR)

    #TODO add NN to detect eyes and only search on this areas

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    (lower, upper) = boundariesEyeHSV[indexColor]
    lower_blue = np.array([lower[0], lower[1], lower[2]])
    upper_blue = np.array([upper[0], upper[1], upper[2]])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    coloured = frame.copy()
    coloured[mask == 0] = BLACK_BG  #black background

    
    # apply erode & dilate to mask. After detect contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    maskED = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(maskED, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # calculate area of this color in the image
    for c in cnts:
        colorArea[indexColor] += cv2.contourArea(c)
        #cv2.drawContours(frame,[c], 0, (0,0,0), 2)
    totalColorArea = totalColorArea + colorArea[indexColor]
    cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    cv2.imshow("maskED", maskED)
    if indexColor == 0:
        etiqueta = 'naranja'
    elif indexColor == 1:
        etiqueta = 'amarillo'
    elif indexColor == 2:
        etiqueta = 'verde'
    elif indexColor == 3:
        etiqueta = 'ceeste'
    cv2.imshow(etiqueta, coloured)

    key = cv2.waitKey(0)
    if key == 27:
        break
    indexColor = indexColor + 1
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