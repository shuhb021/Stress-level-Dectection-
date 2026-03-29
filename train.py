import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import EmotionCNN

# GPU setup with robust fallback
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Test a small operation to ensure kernels work
        torch.zeros(1).to(device)
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA not available)")
except Exception as e:
    device = torch.device("cpu")
    print(f"CUDA initialization failed, falling back to CPU. Error: {e}")

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

print("Loading and parsing dataset...")
data = pd.read_csv("data211.csv")

# Vectorized pixel parsing: Split space-separated strings and convert to float32
pixels = data['pixels'].values
X = np.array([np.fromstring(p, sep=' ', dtype='float32') for p in pixels])
X = X.reshape(-1, 1, 48, 48) / 255.0

y = data['emotion'].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.long)

X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.long)

train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64)



model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Training loop
epochs = 25

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Average Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(),"stress_model.pth")

print("Training Completed")