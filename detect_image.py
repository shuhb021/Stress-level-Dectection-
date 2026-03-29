import cv2
import torch
import numpy as np
import os

from model import EmotionCNN

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("stress_model.pth", map_location=device))
model.eval()

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Folder containing test images
folder = "test_images"

# Loop through images
for file in os.listdir(folder):

    path = os.path.join(folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for x,y,w,h in faces:

        roi = gray[y:y+h, x:x+w]

        roi = cv2.resize(roi,(48,48))
        roi = roi/255.0

        roi = torch.tensor(roi).unsqueeze(0).unsqueeze(0).float().to(device)

        pred = model(roi)

        emotion = emotion_labels[pred.argmax().item()]

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(img,emotion,(x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Result", img)

    cv2.waitKey(0)

cv2.destroyAllWindows()