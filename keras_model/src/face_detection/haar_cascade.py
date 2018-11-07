import numpy as np
import cv2
import matplotlib.pyplot as plt 


class Haar_cascade:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def get_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        face = np.zeros((32,32))
        biggest = 0
        for (x,y,w,h) in faces_detected:
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            if h+w > biggest:
                roi_color = image[y:y+h, x:x+w]    
                face = gray[y:y+h, x:x+w]
                biggest = h + w
        
        if(biggest == 0):
            print("no face found!!")

        return face, image




def main():
    img = cv2.imread('image/test6.jpg')

    face, region = Haar_cascade().get_face(img)
    
    cv2.imshow('image',face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image',region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
