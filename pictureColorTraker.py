import cv2
import sys
import os
import numpy as np

def nothing(x):
    pass

# default index
index = 1
prevIndex = 1
boundariesHSV = [
    ([8,0,50],[20,255,255]),   #naranja HUE 0-179; Sat 0-255; Val 0-255
    ([21,0,50],[30,255,255]),   #amarillo HUE 0-179; Sat 0-255; Val 0-255
    ([31,0,50],[65,255,255]),   #verde HUE 0-179; Sat 0-255; Val 0-255
    ([80,0,50],[115,255,255]),  #celeste
]

if len(sys.argv) < 2:
    print('\nUsage: pictureColorTraker input_color_folder')
    print('\tCOLOR_OPTION: (color) | grey')
    print('\tRESIZE_IMAGE: (true) | false')
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

(lower, upper) = boundariesHSV[indexColor]
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