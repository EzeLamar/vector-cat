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
boundariesSpotsHSV = {
    'blanco' : ([0,0,162],[179,51,255]),       #blanco        [0]  listo
    'gris' : ([27,0,0],[179,76,140]),       #gris          [1]    listo
    'naranjaClaro' : ([10,60,108],[26,171,255]),     #naranjaClaro  [2]  listo
    'naranjaOscuro' : ([10,130,125],[15,255,200]),   #naranjaOscuro [3]   listo
    'marron' : ([0,0,0],[179,76,69]),           #marron        [4]  DESCARTADO
    'negro' : ([0,32,0],[179,255,69]),         #negro         [5]  listo
}

FaceColours = {
    'blanco': [255,255,255],      #blanco        [0]
    'gris': [200,200,200],       #gris          [1]
    'naranjaClaro': [84,168,255],    #naranjaClaro  [2]
    'naranjaOscuro': [0,56,112],   #naranjaOscuro [3]
    'marron': [0,0,0],    #marron        [4]    DESCARTADO, lo uso como negro
    'negro': [0,0,0],      #negro         [5]
}

NameColours = [
    'blanco',
    'gris',
    'naranjaClaro',
    'naranjaOscuro',
    'marron',      #DESCARTADO DE LOS COLORES
    'negro',
]

def extractSpotFromFrame(originalFrame, colourToFilter):
    # frame to return colour extraction
    black_frame = np.zeros(originalFrame.shape,np.uint8)
    black_frame[:,:]= PINK_BG
    finalFaceMask = black_frame.copy()

    # change colout format of original frame
    hsv = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2HSV)

    # get value boundaries to detect the specific colour from original frame
    (lower, upper) = boundariesSpotsHSV[colourToFilter]
    l_h = lower[0]
    l_s = lower[1]
    l_v = lower[2]
    u_h = upper[0]
    u_s = upper[1]
    u_v = upper[2]
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    coloured = originalFrame.copy()
    coloured[mask == 0] = PINK_BG  #pink background

    # apply erode & dilate to mask. After detect contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    maskED = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(maskED, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # apply mask in finalMask with their colour
    finalFaceMask[np.where((maskED == [255] ))] = FaceColours[colourToFilter]

    # calculate area of this color in the image
    totalArea = np.size(originalFrame, 0) * np.size(originalFrame, 1)
    colorArea = 0
    for c in cnts:
        colorArea += cv2.contourArea(c)
    percentageArea = (colorArea*100)/totalArea
    print('\t'+colourToFilter+' Area:'+str(percentageArea)+'%')

    cv2.imshow("originalFrame", originalFrame)
    #cv2.imshow("mask", mask)
    cv2.imshow("maskED", maskED)
    #cv2.imshow(colourToFilter, coloured)
    cv2.imshow(colourToFilter+"_filter", finalFaceMask)
    return maskED

def generateAllColoursMask():
    frame = cv2.imread('../../media/input/spots/negro/cat5.jpg',cv2.IMREAD_COLOR)
    # finalMask
    black_frame = np.zeros(frame.shape,np.uint8)
    black_frame[:,:]= PINK_BG
    finalFaceMask = black_frame.copy()

    for colour in NameColours:
        finalFaceMask[np.where((extractSpotFromFrame(frame,colour) == [255] ))] = FaceColours[colour]
        cv2.imshow('finalMask', finalFaceMask)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

generateAllColoursMask()