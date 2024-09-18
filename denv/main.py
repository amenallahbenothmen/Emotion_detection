import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage
from datetime import datetime, timedelta
from deepface import DeepFace
import subprocess


subprocess.run(["python", "adddata.py"])

cred = credentials.Certificate("face-recognition-1603f-firebase-adminsdk-pn2v7-578c875d61.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-1603f-default-rtdb.europe-west1.firebasedatabase.app/",
    'storageBucket': "face-recognition-1603f.appspot.com"
})

bucket = storage.bucket()

# Initialize the video capture object
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("../Resources/background.png")
# Importing modes
folderModePath = '../Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

file = open('EncodeFile.p', 'rb')
encodeListKnownwithIds = pickle.load(file)
file.close()

encodeListKnown, usersIds = encodeListKnownwithIds
print("Encode file loaded")
print(usersIds)

# Initialize mode and timing variables
modeType = 0
id = -1
imgUser = []
lastAttendanceTime = None
modeSwitchTime = None
displayMode1Time = None

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # 1/4
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Face recognition
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                id = usersIds[matchIndex]

                if modeType == 0:
                    # Switch to Mode 1 and fetch user data
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    modeType = 1
                    modeSwitchTime = datetime.now()
                    displayMode1Time = datetime.now()

                    # Get user data
                    userinfo = db.reference(f'Users/{id}').get()  # Fetch user data
                    blob = bucket.get_blob(f'Images/{id}.png')
                    if blob is None:
                        print(f"Blob not found for ID: {id}")
                        continue  # Skip to the next frame
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgUser = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    lastAttendanceTime = datetime.strptime(userinfo.get('Last_attendence_time', "1970-01-01 00:00:00"),
                                                           "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - lastAttendanceTime).total_seconds()

                    if secondsElapsed > 60:
                        # Detect emotion using the original image
                        try:
                            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                            emotion = result[0]['dominant_emotion']
                        except ValueError as e:
                            print(f"Error detecting emotion: {e}")
                            emotion = "N/A"
                        
                        # Update Firebase database
                        ref = db.reference(f'Users/{id}')
                        userinfo['Total_attendence'] += 1
                        userinfo['Emotion'] = emotion  # Update the emotion in userinfo
                        ref.child('Total_attendence').set(userinfo['Total_attendence'])
                        ref.child('Last_attendence_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        ref.child('Emotion').set(emotion)
                        
                        # Log attendance time and emotion

                        log_date = datetime.now().strftime("%Y-%m-%d")
                        log_ref = db.reference(f'Logs/{id}/{log_date}')
                        log_ref.push({
                            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'emotion': emotion
                        })

                    else:
                        modeType = 3
                        modeSwitchTime = datetime.now()  # Update modeSwitchTime for Mode 3

    if modeType == 1:
        # Display Mode 1 for 5 seconds
        if datetime.now() - displayMode1Time >= timedelta(seconds=5):
            modeType = 2
            modeSwitchTime = datetime.now()

    elif modeType == 2:
        # Display Mode 2 for 5 seconds
        if datetime.now() - modeSwitchTime >= timedelta(seconds=5):
            modeType = 0
            imgUser = []

    elif modeType == 3:
        # Display Mode 3 for 3 seconds and then switch to Mode 0
        if datetime.now() - modeSwitchTime >= timedelta(seconds=3):
            modeType = 0

    # Update background based on the mode
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if modeType == 1:
        # Show user info and image only in Mode 1
        if imgUser is not None and imgUser.size > 0:
            target_size = (216, 216)
            imgUser_resized = cv2.resize(imgUser, target_size)
            imgBackground[175:175 + 216, 909:909 + 216] = imgUser_resized

        cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(userinfo.get('Emotion', 'N/A')), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(userinfo.get('Total_attendence', 'N/A')), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
