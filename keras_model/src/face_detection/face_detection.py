import numpy as np
import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

img = cv2.imread('hk_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5) 
faces = []
for (x,y,w,h) in faces_detected:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]    
    faces.append(roi_gray)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i, face in enumerate(faces):
    cv2.imshow('face',face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('face_image/face{0}.jpeg'.format(i), face)
