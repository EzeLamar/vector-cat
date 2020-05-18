import sys
sys.path.insert(1, '../')
from model import *
import cv2
import os
import numpy as np

#constants
BLACK_BG = (0, 0, 0)
WHITE_COLOR = (255,255,255)
PATH_MODAL = '../../../model/prueba100fotos_inv_2.sh'
PATH_CASCADE = '../../../cascade/haarcascade_frontalcatface.xml'
DOTS = [
    'inEye_L',
    'outEye_L',
    'upEye_L',
    'botEye_L',
    'holeNose_L',
    'botNose',
    'outEar:L',
    'upCheek_L',
    'botCheek_L',
    'botMouth',
    'inEye_R',
    'outEye_R',
    'upEye_R',
    'botEye_R',
    'holeNose_R',
    'botCheek_R',
    'outEar_R',
    'upNose_L',
    'upNose_R',
    'upCheek_R',
    'centerMouth',
]


if len(sys.argv) != 2:
    print('\nUsage: getImageAndDefaultDots input_image')
    print('\tinput_image: name of image to crop & et dots; example: "image1"')
    exit()

# load image to scan..
actualFolder = 'test1'    #sys.argv[1]
actualImage = sys.argv[1]
pathMedia = '../../../media/trainingImages/'+actualFolder
picturePath = pathMedia+'/input/'+actualImage
outputPath = pathMedia+'/output/'

# Load the model built in the previous step
model = load(os.path.abspath(PATH_MODAL))

# Face cascade to detect faces
face_cascade = cv2.CascadeClassifier(os.path.abspath(PATH_CASCADE))

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

    # get the image on grayScale in 96x96 size
    croppedFace = gray[y:y+h, x:x+w]
    croppedFace = cv2.resize(croppedFace, (96, 96), interpolation = cv2.INTER_AREA)

    # Map the Keypoints back to the original image
    face_resized_color = cv2.resize(color_crop_face, (96, 96), interpolation = cv2.INTER_AREA)

    # Pair them together
    for i, co in enumerate(keypoints[0][0::2]):
        points.append((co, keypoints[0][1::2][i]))

    # Add KEYPOINTS to the original
    i = 0
    with open(os.path.join(outputPath, 'image{}_meta.mtl'.format(actualImage.split('.')[0])), 'w') as outfile:
        for keypoint in points[0:21]:
            newLine = DOTS[i]+';'+str(keypoint[0]/96)+';'+str(1-keypoint[1]/96)
            outfile.write(newLine.replace('.',',') + '\n')
            cv2.circle(face_resized_color, keypoint, 1, (0,255,0), 1)
            i += 1
    originalDots[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

    # show face detected with dots
    cv2.imshow('original',original)
    cv2.imshow('cropped',croppedFace)
    cv2.imshow("face detected", originalDots)

    # save image in greyscale for training
    cv2.imwrite(os.path.join(outputPath, 'image{}.png'.format(actualImage.split('.')[0])), croppedFace)
    # save default dots for adjust & training
    # TODO

    # wait to press some button and close
    cv2.waitKey(0)
    cv2.destroyAllWindows()