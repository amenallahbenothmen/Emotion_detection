import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage
import os
import cv2
import pickle
import face_recognition
import random

# Initialize Firebase
cred = credentials.Certificate("face-recognition-1603f-firebase-adminsdk-pn2v7-578c875d61.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-1603f-default-rtdb.europe-west1.firebasedatabase.app/",
    'storageBucket': "face-recognition-1603f.appspot.com"
})

ref = db.reference("Users")
bucket = storage.bucket()

# Path to the folder containing user images
folderPath = '../Images'
PathList = os.listdir(folderPath)

# Fetch existing user IDs from the database
existing_users = ref.get() if ref.get() is not None else {}
existing_user_ids = set(existing_users.keys())

# Lists to hold images and user IDs
imgList = []
allUserIds = []  # For both new and old users
newUserIds = set()  # Only new user IDs

def generate_unique_id(existing_ids):
    while True:
        new_id = str(random.randint(100000000, 999999999))
        if new_id not in existing_ids:
            return new_id

for path in PathList:
    user_id = os.path.splitext(path)[0]
    local_file_path = os.path.join(folderPath, path)

    if not (user_id.isdigit() and len(user_id) == 9):
        # Generate a unique 9-digit ID
        user_id = generate_unique_id(existing_user_ids)
        new_file_path = os.path.join(folderPath, f"{user_id}.png")
        os.rename(local_file_path, new_file_path)
        local_file_path = new_file_path

        data = {
            "Emotion": "N/A",
            "Total_attendence": 0,
            "Last_attendence_time": "1970-01-01 00:00:00"  # Default value for non-existent attendance time
        }
        ref.child(user_id).set(data)
        print(f"Added new user {user_id} to the database.")

        # Upload image to Firebase Storage
        firebase_file_path = f'Images/{user_id}.png'
        blob = bucket.blob(firebase_file_path)
        blob.upload_from_filename(local_file_path)
        print(f"Uploaded image for user {user_id} to Firebase Storage.")

        newUserIds.add(user_id)
    
    # Add image to list for encoding
    imgList.append(cv2.imread(local_file_path))
    allUserIds.append(user_id)

# Regenerate encodings only if there are new users
if newUserIds:
    def findEncodings(imagesList):
        encodeList = []
        if imagesList:  # Check if there are any images
            for img in imagesList:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img)
                if encodings:  # Check if encodings are found
                    encodeList.append(encodings[0])
        return encodeList

    # Generate encodings for all images
    encodeListKnownwithIds = [findEncodings(imgList), allUserIds]

    # Save encodings to file
    with open("EncodeFile.p", 'wb') as file:
        pickle.dump(encodeListKnownwithIds, file)

    # Upload EncodeFile.p to Firebase Storage
    blob = bucket.blob('EncodeFile.p')
    blob.upload_from_filename('EncodeFile.p')
    print('Encode file generated and uploaded to Firebase Storage')
else:
    print("No new users found. Encode file not modified.")
