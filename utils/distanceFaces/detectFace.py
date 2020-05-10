from model import *
import cv2
import numpy as np
import math
import os
import sys

# To find the euclidean distance between the two embeddings
def euclideanDistance(array1, array2, cantDots):
    sumDistances = 0
    for i in range(0, cantDots-1):
        sumDistances = np.sum(np.square(array1[i] - array2[i]))
    return math.sqrt(sumDistances)

def euclideanDistance2(arrayOfDots, cantDots):
    sumDistances = 0.0
    for firstPos in range(0, cantDots-2):
        for secondPos in range(firstPos+1, cantDots-1):
            distanceTwoDots = np.square(arrayOfDots[firstPos] - arrayOfDots[secondPos])
            sumDistances += distanceTwoDots
    return math.sqrt(sumDistances)

def euclideanDistance3(arrayOfDots, cantDots):
    sumDistances = 0.0
    for firstPos in range(0, cantDots-2):
        distanceTwoDots = float(np.square(arrayOfDots[firstPos] - arrayOfDots[cantDots-1]))
        sumDistances += float(distanceTwoDots)
    return float(math.sqrt(sumDistances))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: test <tflite_model> <haar_cascade_classifier>')

    # Load the model built in the previous step
    model = load(os.path.abspath(sys.argv[1]))
    path = os.path.abspath(sys.argv[1])

    # Face cascade to detect faces
    face_cascade = cv2.CascadeClassifier(os.path.abspath(sys.argv[2]))

    # Load the video
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print('--(!)Error opening WebCam')
        exit()

    # Keep looping
    while True:
        ret, frame = camera.read()
        if not ret: break

        # Process current frame
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.25, 6)

        for (x, y, w, h) in faces:
            # Grab the face
            gray_face = gray[y:y+h, x:x+w]
            color_face = frame[y:y+h, x:x+w]

            # Normalize to match the input format of the model - Range of pixel to [0, 1]
            gray_normalized = gray_face / 255

            # Resize it to 96x96 to match the input format of the model
            original_shape = gray_face.shape # A Copy for future reference
            face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized = face_resized.reshape(1, 96, 96, 1)

            # Predicting the keypoints using the model
            keypoints = model.predict(face_resized)
            # De-Normalize the keypoints values
            keypoints = keypoints * 48 + 48  

            # Map the Keypoints back to the original image
            face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)

            # Pair them together
            points = []
            for i, co in enumerate(keypoints[0][0::2]):
                points.append((co, keypoints[0][1::2][i]))

            #show KEYPOINTS and his ID
            #print(keypoints)
            keypointsEze2 = [65.62021, 39.646862, 30.841415, 36.879356, 59.5819, 39.93722,  72.04867, 41.283627, 37.901814, 39.800255, 24.816414, 38.72844, 56.50785,  33.405674, 78.42453,  33.535557, 40.908592, 31.424381, 19.17184,  29.101841, 45.85429, 61.51329,  60.900276, 77.113235, 32.189926, 75.00681, 46.092793, 73.83165, 45.340206, 83.2624, 87.647415, 38.255924, 11.773563, 34.441895] 
            keypointsEze = [65.65066,  38.738087, 30.640045, 37.305893, 59.498592, 39.154102, 71.937355,
 39.916416, 37.049095, 38.995533, 23.991869, 38.739784, 56.345665, 31.304502,
 77.93844,  31.992832, 39.91857,  30.557163, 18.280434, 31.099487, 47.44179,
 57.56422,  61.281445, 76.51394,  33.61339,  75.4884,   47.30845,  71.87795,
 46.878456, 83.36195,  86.07471,  38.271065,  9.431927, 37.153084]
            IDPersona = cv2.norm(keypoints)
            #####################################falta recorrer el arreglo y comparar los valores, sumando las distancias 
            #diferencia = euclideanDistance(keypointsEze, keypoints[0], 33)
            distanciasEze = euclideanDistance3 (keypointsEze, 33)
            distanciaActual = euclideanDistance3 (keypoints[0], 33)
            diferencia = abs(distanciasEze - distanciaActual)
            #print('distanciasEze: '+ str(distanciasEze))
            #print('distanciasActual: '+ str(distanciaActual))
            #print('diferencia:'+ str(diferencia))
            
            #print(keypoints[0])
            if diferencia < 0.6:
                print('Hola Ezequiel')

            # Add KEYPOINTS to the frame
            for keypoint in points:
                cv2.circle(face_resized_color, keypoint, 1, (0,255,0), 1)

            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

        # Display the resulting frame
        cv2.imshow("Facial Keypoints", frame)

        # Press ESC on keyboard to exit
        if cv2.waitKey(1) == 27: break

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()