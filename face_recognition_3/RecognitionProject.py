import cv2
import numpy as np
import face_recognition
import os
import pickle

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

# Load the encodings done before
encodings_file = "encodings.pickle"
data = pickle.loads(open(encodings_file, "rb").read())

imgTest = face_recognition.load_image_file('TestImages/Arnold_Schwarzenegger.jpg')
imgTest = cv2.resize(imgTest, (0, 0), None, 0.25, 0.25)
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#Locating the face in the image
faceLocTest = face_recognition.face_locations(imgTest)
encodeTest = face_recognition.face_encodings(imgTest, faceLocTest)

cv2.rectangle(imgTest, (faceLocTest[0][3], faceLocTest[0][0]), (faceLocTest[0][1], faceLocTest[0][2]), (0, 255, 0), 2)

for encodeFace, faceLoc in zip(encodeTest, faceLocTest):
  # Comparing the image with the database
  matches = face_recognition.compare_faces(data["encodings"], encodeFace)
  # Descobrindo a distância entre as duas imagens, quanto menor d, mais provável serem a mesma pessoa
  faceDis = face_recognition.face_distance(data["encodings"], encodeFace)
  print(faceDis)
  print(knownClasses)

  matchIndex = np.argmin(faceDis)
  print(matchIndex)

  if matches[matchIndex]:
    name = classNames[matchIndex]
    print(name)
    cv2.putText(imgTest, f'{name}\n{round(faceDis[matchIndex],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2_imshow(imgTest)

