import cv2
import torch
import numpy as np
import time

from model import EmotionCNN

# Emotion labels and modern color palette (BGR)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 0),    # Dark Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Cyan/Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 191, 0), # Light Blue
    'Neutral': (200, 200, 200) # Gray
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
print(f"Using device: {device}")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load("stress_model.pth", map_location=device))
model.eval()

# UI Helper: Draw stylish corner brackets
def draw_corners(img, x, y, w, h, color, thickness=2, length=20):
    # Top Left
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    # Top Right
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom Left
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom Right
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

cap = cv2.VideoCapture(0)
prev_time = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Dashboard Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"AI STRESS DETECTION | FPS: {int(fps)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"DEVICE: {device} ({gpu_name})", (frame.shape[1] - 350, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = torch.tensor(roi).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(roi)
        
        emotion_idx = pred.argmax().item()
        emotion = emotion_labels[emotion_idx]
        color = emotion_colors.get(emotion, (0, 255, 0))

        # Dynamic UI Elements
        draw_corners(frame, x, y, w, h, color, thickness=3)
        
        # Label Background Box
        (text_w, text_h), baseline = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w, y), color, -1)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Stress Detection",frame)

    if cv2.waitKey(1)==ord("q"):

        break

cap.release()
cv2.destroyAllWindows()