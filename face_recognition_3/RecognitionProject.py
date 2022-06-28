import cv2
import numpy as np
import face_recognition
import os

path = "ImagesProject"
images = []
classNames = []
myList = os.listdir(path)
imgPaths = myList

for img_path in imgPaths:
    imgFolder = os.listdir(f'{path}/{img_path}')
    for imgfile in imgFolder:
        curImg = cv2.imread(f'{path}/{img_path}/{imgfile}')
        images.append(curImg)
        classNames.append(img_path)

print(len(images))


def find_encodings(images):
    encodeList = []
    knownClasses = []
    i = 0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            knownClasses.append(classNames[i])
            print(i)
            i += 1
        except:
            pass
    return encodeList, knownClasses

encodeListKnown, knownClasses = find_encodings(images)
print(knownClasses)


# Testing with a unknown image
imgTest = face_recognition.load_image_file('TestImages/paris_hilton.jpeg')
imgTest = cv2.resize(imgTest, (0, 0), None, 0.25, 0.25)
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
encodeTest = face_recognition.face_encodings(imgTest)[0]

# Comparing the image with the database
results = face_recognition.compare_faces(encodeListKnown, encodeTest)

for i in range(len(results)):
    if results[i] == True:
        print(i)
        print(knownClasses[i])
# print(results.count(True))

