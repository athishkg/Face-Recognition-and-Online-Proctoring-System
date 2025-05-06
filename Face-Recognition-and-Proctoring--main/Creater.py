import cv2
import subprocess

load = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

name = input("Enter Username : ")
id = input("Enter Unique id (maxLen = 4) : ")

f = open("datatext.txt", "a")
f.write(str(id) + " " + name + "\n")
val = 0

while True:
    status, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = load.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        val += 1
        cv2.imwrite("Data/" + str(id) + "." + str(val) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(70)

    cv2.imshow('FaceDetect', img)
    cv2.waitKey(1)
    if val >= 50:
        break

f.close()
cap.release()
cv2.destroyAllWindows()
subprocess.run(['python', 'trainer.py'])