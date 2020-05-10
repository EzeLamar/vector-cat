from model import *
import cv2
import numpy as np
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: test <tflite_model> <haar_cascade_classifier>')

    # Load the model built in the previous step
    model = load(os.path.abspath(sys.argv[1]))

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

            # draw a blue rectangle on the image
            x -= 10
            y -= 10
            w += 20
            h += 20
            if x < 0:
                x = 0
            if y < 0:
                y = 0

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

            # Add KEYPOINTS to the frame
            for keypoint in points[0:21]:
                cv2.circle(face_resized_color, keypoint, 1, (0,255,0), 1)

            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

        # Display the resulting frame
        cv2.imshow("Facial Keypoints", frame)

        # Press ESC on keyboard to exit
        if cv2.waitKey(1) == 27: break

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
