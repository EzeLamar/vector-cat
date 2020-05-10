import cv2
import sys
import os
import numpy as np

def nothing(x):
    pass

#constants
PINK_BG = (120, 0, 255)
BLACK_BG = (0, 0, 0)
PINK_BG = (255,255,0)
WHITE_BG = (255, 255, 255)

# default index
index = 1
prevIndex = 1
boundariesEyeHSV = [
    ([0,0,137],[179,72,255]),       #blanco        [0]  listo
    ([0,0,0],[179,72,255]),       #gris          [1]
    ([10,60,0],[26,171,255]),     #naranjaClaro  [2]
    ([10,130,125],[14,255,176]),   #naranjaOscuro [3]
    ([0,0,0],[179,62,69]),          #marron        [4]
    ([0,0,0],[179,255,69]),         #negro         [5]  listo
]

if len(sys.argv) < 2:
    print('\nUsage: colorEyeCalibrator INPUT_COLOR_FOLDER')
    print('\tINPUT_COLOR_FOLDER: blanco | gris | marron | naranja | negro')
    print('\tCONTROLS: \n\t\tn key: next image\n\t\tp key: previous image\n\t\tesc: close program')
    exit()

# load image file to work..
actualFolder = sys.argv[1]
pictureFolder = '../../media/input/spots/'+actualFolder

cv2.namedWindow("toolbars")

if (actualFolder == 'blanco'):
    indexColor = 0
elif (actualFolder == 'gris'):
    indexColor = 1
elif (actualFolder == 'naranjaClaro'):
    indexColor = 2
elif (actualFolder == 'naranjaOscuro'):
    indexColor = 3
elif (actualFolder == 'marron'):
    indexColor = 4
elif (actualFolder == 'negro'):
    indexColor = 5
else:
    print('Error: color no encontrado.')
    exit()

(lower, upper) = boundariesEyeHSV[indexColor]
cv2.createTrackbar("L – H", "toolbars", lower[0], 179, nothing)
cv2.createTrackbar("L – S", "toolbars", lower[1], 255, nothing)
cv2.createTrackbar("L – V", "toolbars", lower[2], 255, nothing)
cv2.createTrackbar("U – H", "toolbars", upper[0], 179, nothing)
cv2.createTrackbar("U – S", "toolbars", upper[1], 255, nothing)
cv2.createTrackbar("U – V", "toolbars", upper[2], 255, nothing)

while True:
    frame = cv2.imread(pictureFolder+'/cat'+str(index)+'.jpg',cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L – H", "toolbars")
    l_s = cv2.getTrackbarPos("L – S", "toolbars")
    l_v = cv2.getTrackbarPos("L – V", "toolbars")
    u_h = cv2.getTrackbarPos("U – H", "toolbars")
    u_s = cv2.getTrackbarPos("U – S", "toolbars")
    u_v = cv2.getTrackbarPos("U – V", "toolbars")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    coloured = frame.copy()
    coloured[mask == 0] = PINK_BG  #pink background

    
    # apply erode & dilate to mask. After detect contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    maskED = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=0)
    cnts = cv2.findContours(maskED, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # calculate area of this color in the image
    totalArea = np.size(frame, 0) * np.size(frame, 1)
    colorArea = 0
    for c in cnts:
        colorArea += cv2.contourArea(c)
        cv2.drawContours(frame,[c], 0, (0,0,0), 2)
    percentageArea = (colorArea*100)/totalArea
    print('color Area:'+str(percentageArea)+'%')


    cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    cv2.imshow("maskED", maskED)
    cv2.imshow("result",coloured)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 115:          #char 's'
        break
    elif key == 110:        #char 'n'
        index = index + 1
    elif key == 112:        #char 'p'
        index = index - 1
cv2.destroyAllWindows()