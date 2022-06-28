import cv2
import numpy as np
import face_recognition

imgAaron = face_recognition.load_image_file('ImagesBasic/Aaron_Eckhart/Aaron_Eckhart_0001_0000.jpg')
imgAaron = cv2.cvtColor(imgAaron, cv2.COLOR_BGR2RGB)

cv2.imshow('Aaron Eckhart', imgAaron)
cv2.waitKey(0)