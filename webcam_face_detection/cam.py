from pyimagesearch.facedetector import FaceDetector
from pyimagesearch import imutils
import argparse
import cv2
import numpy as np

ap  = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True, help="cascade face detection path")
ap.add_argument('-v', '--video', help='video path')

args = vars(ap.parse_args())

fd = FaceDetector(args['face'])

if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])


while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    frameCopy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faceRects = fd.detect(gray, scaleFactor = 1.1, 
                        minNeighbors = 5, minSize = (30, 30))
    
    for (x, y, w, h) in faceRects:
        cv2.rectangle(frameCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    frameCopy = imutils.flip(frameCopy)

    cv2.imshow("Face", frameCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()