import cv2  # for computer vision and recording the video
import numpy as np  # to give a numerical value to the face distance
import face_recognition  # description of a captured face
import os  # to locate the images of the users
from datetime import datetime  # to record the date and time of entry

path = 'images'  # path of the images
images = []
personNames = []
myList = os.listdir(path)  # locating the path
# print(myList)

# to read the images for the specified path
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')  # reading the images
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])


# print(personNames)

# encoding the images to recognise from computer vision
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# to enter the user details in the .CSV file
def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


# to video record the attendance
filename = 'attendance.avi'
frames_per_second = 20.0
my_res = '480p'

# resolution adjustment
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# types of pixel adjustments
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

# using the dimensions
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current capture device
    # to the resulting resolution
    change_res(cap, width, height)
    return width, height

# using the XVID extension for creating the video extension
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID')
}

# to save the recorded video
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# scan the encodings of the facial features before scanning through computer vision
encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

# computer vision (camera)
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 15, get_dims(cap, my_res))

# creating a green frame around the detected face.
while True:
    ret, frame = cap.read()
    out.write(frame)
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        # frame
        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
