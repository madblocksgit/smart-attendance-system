import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time



while True: # waiting statement, it will wait for user command
    f=open('log.txt')
    t=f.read()
    if t=='start':
        break

# from PIL import ImageGrab

path = 'final'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread('final/' + cl)
    #print(curImg)
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)
#print(images)

def findEncodings(images):
    encodeList = []
    #print(images)

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
 with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(1)
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
size=(frame_width,frame_height)
date_string = time.strftime("%Y-%m-%d-%H-%M-%S")
result=cv2.VideoWriter('videos/'+str(date_string)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,size)
while True:
    f=open('log.txt')
    if(f.read()=='stop'):
        break
    success, img = cap.read()
    result.write(img)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            date_string = time.strftime("%Y-%m-%d-%H-%M-%S")
            cv2.imwrite('known/'+str(date_string)+'.jpg',img)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            print('Unknown Face')
            cv2.imwrite('test.jpg',img)
            date_string = time.strftime("%Y-%m-%d-%H-%M-%S")
            cv2.imwrite('intruders/'+str(date_string)+'.jpg',img)
        
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()
cap.release()
result.release()
os.system('python madhu.py')