import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as transforms
import pandas as pd
import cv2
import os
import mediapipe as mp

def extract_hand_landmarks(frame):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.6)
    results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_hands.close()
    return results

def extract_pose_landmarks(frame):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.6)
    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_pose.close()
    return results

def extract_face_landmarks(frame):
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.6)
    results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_face.close()
    return results

class IterDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
    
# Define the 3D CNN model
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(64 * 8 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

        self.hand_fc = nn.Linear(3, 256)
        self.pose_fc = nn.Linear(3, 256)
        self.face_fc = nn.Linear(3, 256)
        self.final_fc = nn.Linear(256 * 4, 1)
        self.dropout = nn.Dropout(0.5)


    def forward(self, frames, hand_landmarks, pose_landmarks, face_landmarks):
        # Process frames
        x = self.pool(torch.relu(self.conv1(frames)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        # Process hand landmarks
        hand_x = torch.relu(self.hand_fc(hand_landmarks))
        hand_x = self.dropout(hand_x)

        # Process pose landmarks
        pose_x = torch.relu(self.pose_fc(pose_landmarks))
        pose_x = self.dropout(pose_x)

        # Process face landmarks
        face_x = torch.relu(self.face_fc(face_landmarks))
        face_x = self.dropout(face_x)

        # Concatenate all the features
        x = torch.cat((x, hand_x, pose_x, face_x), dim=1)
        x = torch.relu(self.final_fc(x))

        return x

# Define custom dataset for ASL videos
class ASLDataset(Dataset):
    def __init__(self, csv_file, video_folder, transform=None):
        self.df = pd.read_csv(csv_file, sep= "\t", header= 0)
        self.video_folder = video_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_name = self.df.loc[idx, 'VIDEO_NAME']
        start_time = self.df.loc[idx, 'START_REALIGNED']
        end_time = self.df.loc[idx, 'END_REALIGNED']
        sentence = self.df.loc[idx, 'SENTENCE']
        transformed_frames = []

        video_path = os.path.join(self.video_folder, video_name)
        frames = self.extract_frames(video_path, start_time, end_time)

        hand_landmarks = [self.extract_hand_landmarks(frame) for frame in frames]
        pose_landmarks = [self.extract_pose_landmarks(frame) for frame in frames]
        face_landmarks = [self.extract_face_landmarks(frame) for frame in frames]


        hand_landmarks = [torch.tensor(lm).view(-1, self.hand_landmark_dim) for lm in hand_landmarks]
        pose_landmarks = [torch.tensor(lm).view(-1, self.pose_landmark_dim) for lm in pose_landmarks]
        face_landmarks = [torch.tensor(lm).view(-1, self.face_landmark_dim) for lm in face_landmarks]

        if self.transform:
            transformed_frames.append(self.transform(frame) for frame in frames)

        return transformed_frames, hand_landmarks, pose_landmarks, face_landmarks, sentence


    def extract_frames(self, video_path, start_time, end_time):
        # Use OpenCV to extract frames from the video based on start and end time
        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        frames = []
        frame_count = 0

        while True:
            ret, frame = video.read()

            if not ret:
                break

            if frame_count >= start_frame and frame_count <= end_frame:
                frames.append(frame)

            if frame_count > end_frame:
                break

            frame_count += 1

        video.release()

        return frames

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_classes = 26
batch_size = 16
epochs = 10
learning_rate = 0.001

# Define transformations for input frames
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

# Load the training dataset
csv_file = '/mnt/d/Project-6367/Dataset/raw_data/how2sign_realigned_train.csv'
video_folder = '/mnt/d/Project-6367/Dataset/raw_videos/'
dataset = ASLDataset(csv_file, video_folder, transform=transform)
gen = IterDataset(dataset)

# Create DataLoader for batch processing
dataloader = DataLoader(gen, batch_size=batch_size)


# Create the model and move it to the device
model = CNN3D().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0

    for frames, hand_landmarks, pose_landmarks, face_landmarks, labels in dataloader:
        
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(frames, hand_landmarks, pose_landmarks, face_landmarks)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'asl_model.pth')
