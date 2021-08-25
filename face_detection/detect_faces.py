from pyimagesearch.facedetector import FaceDetector
import cv2 
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-f", "--face", required=True,
                help="Path to where the face cascade resides")

ap.add_argument("-i", "--image", required=True,
                help="Path to image file")

args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fd = FaceDetector(args['face'])
ed = FaceDetector(r"C:\Users\gyon\Desktop\Case Studies\face_detection\cascades\haarcascade_eye.xml")
faceRects = fd.detect(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
eyeRects = ed.detect(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

'''for (x, y, w, h) in faceRects:
    for (x2, y2, w2, h2) in eyeRects:
        if (x2 > x and x2 < x+h) and (y2 > y and y2 < y+h) :
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
'''
for (x, y, w, h) in faceRects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

for (x, y, w, h) in eyeRects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)