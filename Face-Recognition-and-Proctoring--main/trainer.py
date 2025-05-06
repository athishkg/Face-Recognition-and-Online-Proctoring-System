import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'Data'

def img(path):
    imgpaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]  # getting images path
    # print imgpaths

    faces = []
    users = []
    for imgpath in imgpaths:
        filename = os.path.basename(imgpath)
        try:
            user = int(filename.split('.')[0])  # getting user ID from filename
        except ValueError:
            print("Invalid filename format:", filename)
            continue  # skip this image
        faceimg = Image.open(imgpath).convert('L')  # converting to grayscale
        facenp = np.array(faceimg, 'uint8')
        faces.append(facenp)
        users.append(user)
        cv2.waitKey(10)
    return users, faces

users, faces = img(path)
recognizer.train(faces, np.array(users))  # training
recognizer.save('recognizer/TraningData.yml')  # saving histogram data
cv2.destroyAllWindows()
