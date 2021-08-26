import sys
sys.path.append(r"C:\Users\gyon\Desktop\Case Studies\utils")

import imutils
import argparse
import cv2
import numpy as np

ap  = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='video path')

args = vars(ap.parse_args())

blueLower = np.array([100, 67, 0], dtype="uint8")
blueUpper = np.array([255, 128, 50], dtype="uint8")

if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    
    frameCopy = frame.copy()

    frameCopy = imutils.colorTrack(frameCopy, [100, 67, 0], [255, 128, 50])
    cv2.imshow("Face", frameCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()