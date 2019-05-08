import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture('imagem/teste1.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

size = (
  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('imagem/saida.avi', codec, 23.0, size)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    output.write(img)
        

    #cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()