import cv2
import sys
import os
import numpy as np

def nothing(x):
    pass

#constants
MUST_WAIT = 0
DONT_WAIT = 1
PINK_BG = (120, 0, 255)
BLACK_BG = (0, 0, 0)
PINK_BG = (255,0,255)
WHITE_BG = (255, 255, 255)

# default index
waitingKey = DONT_WAIT
index = 1
prevIndex = 1
boundariesEyeHSV = [
    ([0,0,162],[179,51,255]),       #blanco        [0]  listo
    ([27,0,0],[179,76,140]),       #gris          [1]    listo
    ([10,60,108],[26,171,255]),     #naranjaClaro  [2]  listo
    ([10,130,125],[15,255,200]),   #naranjaOscuro [3]   listo
    ([0,0,0],[179,76,69]),          #marron        [4]  DESCARTADO
    ([0,32,0],[179,255,69]),         #negro         [5]  listo
]

FaceColours = [
    [255,255,255],      #blanco        [0]
    [200,200,200],       #gris          [1]
    [84,168,255],    #naranjaClaro  [2]
    [0,56,112],   #naranjaOscuro [3]
    #[5,33,58],    #marron        [4]    DESCARTADO, lo uso como negro
    [0,0,0],        # reemplazo del marron
    [0,0,0],      #negro         [5]
]


if len(sys.argv) < 2:
    print('\nUsage: colorEyeCalibrator INPUT_COLOR_FOLDER')
    print('\tINPUT_COLOR_FOLDER: blanco | gris | marron | naranja | negro | test <index_image>')
    print('\tCONTROLS: \n\t\tn key: next image\n\t\tp key: previous image\n\t\tesc: close program')
    exit()

# load image file to work..
actualFolder = sys.argv[1]
pictureFolder = '../../media/input/spots/'+actualFolder

cv2.namedWindow("pepe")

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
elif (actualFolder == 'test'):
    indexColor = 0
    waitingKey = MUST_WAIT
    index = sys.argv[2]
else:
    print('Error: color no encontrado.')
    exit()

(lower, upper) = boundariesEyeHSV[indexColor]
cv2.createTrackbar("L – H", "pepe", lower[0], 179, nothing)
cv2.createTrackbar("L – S", "pepe", lower[1], 255, nothing)
cv2.createTrackbar("L – V", "pepe", lower[2], 255, nothing)
cv2.createTrackbar("U – H", "pepe", upper[0], 179, nothing)
cv2.createTrackbar("U – S", "pepe", upper[1], 255, nothing)
cv2.createTrackbar("U – V", "pepe", upper[2], 255, nothing)

frame = cv2.imread(pictureFolder+'/cat'+str(index)+'.jpg',cv2.IMREAD_COLOR)
black_frame = np.zeros(frame.shape,np.uint8)
black_frame[:,:]= PINK_BG
finalFaceMask = black_frame.copy()  # Mascara negra que tendrá el resultado final de los filtros de la cara

while True:
    frame = cv2.imread(pictureFolder+'/cat'+str(index)+'.jpg',cv2.IMREAD_COLOR)
    # black_frame = np.zeros(frame.shape,np.uint8)
    # finalFaceMask = black_frame.copy()  # Mascara negra que tendrá el resultado final de los filtros de la cara
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L – H", "pepe")
    l_s = cv2.getTrackbarPos("L – S", "pepe")
    l_v = cv2.getTrackbarPos("L – V", "pepe")
    u_h = cv2.getTrackbarPos("U – H", "pepe")
    u_s = cv2.getTrackbarPos("U – S", "pepe")
    u_v = cv2.getTrackbarPos("U – V", "pepe")

    actualColor = ''
    if (actualFolder == 'test'):
        if (indexColor == 0):
            actualColor = 'blanco'
        elif (indexColor == 1):
            actualColor = 'gris'
        elif (indexColor == 2):
            actualColor = 'naranjaClaro'
        elif (indexColor == 3):
            actualColor = 'naranjaOscuro'
        elif (indexColor == 4):
            actualColor = 'marron'
        elif (indexColor == 5):
            actualColor = 'negro'

        print('actual color: '+actualColor)
        (lower, upper) = boundariesEyeHSV[indexColor]
        l_h = lower[0]
        l_s = lower[1]
        l_v = lower[2]
        u_h = upper[0]
        u_s = upper[1]
        u_v = upper[2]

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    coloured = frame.copy()
    coloured[mask == 0] = PINK_BG  #pink background

    
    # apply erode & dilate to mask. After detect contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    maskED = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(maskED, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # apply mask in finalMask with their colour
    if (actualFolder == 'test'):
        finalFaceMask[np.where((maskED == [255] ))] = FaceColours[indexColor]

    # calculate area of this color in the image
    totalArea = np.size(frame, 0) * np.size(frame, 1)
    colorArea = 0
    for c in cnts:
        colorArea += cv2.contourArea(c)
        cv2.drawContours(frame,[c], 0, (0,0,0), 2)
    percentageArea = (colorArea*100)/totalArea
    print('\tcolor Area:'+str(percentageArea)+'%')


    cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    cv2.imshow("maskED", maskED)
    cv2.imshow("finalMask", finalFaceMask)
    if (actualFolder == 'test'):
        cv2.imshow(actualColor, coloured)
    else:
        cv2.imshow("result",coloured)

    key = cv2.waitKey(waitingKey)
    if key == 27:
            break
    if (actualFolder == 'test'):
        if key == 110:        #char 'n'
            index = index + 1
            indexColor = 0
        elif key == 112:        #char 'p'
            index = index - 1
            indexColor = 0
        else:
            indexColor += 1
            if indexColor > 5:
                break
    else:
        if key == 110:        #char 'n'
            index = index + 1
        elif key == 112:        #char 'p'
            index = index - 1
cv2.destroyAllWindows()