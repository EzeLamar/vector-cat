import cv2
import sys
import os
import numpy as np

def nothing(x):
    pass

# default index
index = 1
prevIndex = 1
boundariesEyeHSV = [
    ([14,40,95],[19,211,233]),   #naranja HUE 0-179; Sat 0-255; Val 0-255
    ([20,0,50],[30,255,255]),   #amarillo HUE 0-179; Sat 0-255; Val 0-255
    ([31,0,50],[65,255,255]),   #verde HUE 0-179; Sat 0-255; Val 0-255
    ([80,21,50],[115,160,255]),  #celeste
]

if len(sys.argv) < 2:
    print('\nUsage: colorEyeCalibrator INPUT_COLOR_FOLDER')
    print('\tINPUT_COLOR_FOLDER: naranja | amarillo | verde | celeste | test')
    print('\tCONTROLS: \n\t\tn key: next image\n\t\tp key: previous image\n\t\tesc: close program')
    exit()

# load image file to work..
actualFolder = sys.argv[1]
pictureFolder = './media/input/ojos/'+actualFolder

cv2.namedWindow("pepito")

if (actualFolder == 'naranja'):
    indexColor = 0
elif (actualFolder == 'amarillo'):
    indexColor = 1
elif (actualFolder == 'verde'):
    indexColor = 2
elif (actualFolder == 'celeste'):
    indexColor = 3
elif (actualFolder == 'test'):  #modo para testear los colores con el pack de fotos de prueba
    print('modo testeo de los colores')
else:
    print('Error: color no encontrado.')
    exit()

(lower, upper) = boundariesEyeHSV[indexColor]
cv2.createTrackbar("L – H", "pepito", lower[0], 179, nothing)
cv2.createTrackbar("L – S", "pepito", lower[1], 255, nothing)
cv2.createTrackbar("L – V", "pepito", lower[2], 255, nothing)
cv2.createTrackbar("U – H", "pepito", upper[0], 179, nothing)
cv2.createTrackbar("U – S", "pepito", upper[1], 255, nothing)
cv2.createTrackbar("U – V", "pepito", upper[2], 255, nothing)

while True:
    frame = cv2.imread(pictureFolder+'/cat'+str(index)+'.jpg',cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L – H", "pepito")
    l_s = cv2.getTrackbarPos("L – S", "pepito")
    l_v = cv2.getTrackbarPos("L – V", "pepito")
    u_h = cv2.getTrackbarPos("U – H", "pepito")
    u_s = cv2.getTrackbarPos("U – S", "pepito")
    u_v = cv2.getTrackbarPos("U – V", "pepito")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

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