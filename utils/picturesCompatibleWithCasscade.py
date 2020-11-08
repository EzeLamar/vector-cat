# import the opencv library
import cv2
import os
import sys

PATH_OUTPUT_WEBCAM_CASCADE = "../media/output/webcamCascade/"
PATH_CASCADE = '../cascade/haarcascade_frontalcatface.xml'

face_cascade = cv2.CascadeClassifier(os.path.abspath(PATH_CASCADE))
# define a video capture object
vid = cv2.VideoCapture(0)
contador = 1
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    if len(faces) != 0:
        print('encontre cara de gato!')
        cv2.imwrite(os.path.join(PATH_OUTPUT_WEBCAM_CASCADE, 'cat{}.jpg'.format(contador)), frame)
        contador = contador + 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
