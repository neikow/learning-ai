import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from common.utils import get_device


class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()

        # 1x48x48
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 32x48x48
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32x24x24

        # 32x24x24
        self.bn2 = nn.BatchNorm2d(64)
        # 32x24x24
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 64x24x24
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64x12x12

        # 64x12x12
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 128x12x12
        self.bn3 = nn.BatchNorm2d(128)
        # 128x12x12
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128x6x6

        self.flatten = nn.Flatten()

        # 128*6*6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        # 1024
        self.dropout = nn.Dropout(0.3)

        # 1024
        self.fc2 = nn.Linear(1024, 7)  # 7 emotion classes

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 3E-4
DATASET_ROOT = './models'
CSV_FILE = "icml_face_data.csv"
SAVE_PATH = 'models/trained_emotions_model.pth'
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def get_trained_emotions_model() -> EmotionRecognitionModel:
    device = get_device()
    model = EmotionRecognitionModel().to(device)
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Trained model not found at '{SAVE_PATH}'. Please train the model first.")
    model.eval()
    return model


def _train_emotions_model():
    device = get_device()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def get_data_loader(train: bool = True):
        try:
            dataset = torchvision.datasets.FER2013(
                DATASET_ROOT,
                split="train" if train else "test",
                transform=train_transform if train else val_transform,
            )

            return torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True if train else False,
                num_workers=2,
            )
        except RuntimeError as e:
            print(f"\nError loading dataset: {e}")
            print("\nPlease make sure you have done the following:")
            print(f"1. Create a folder named 'fer2013' in this directory.")
            print(f"2. Move your '{CSV_FILE}' into that 'fer2013' folder.")
            print(f"The expected path is: {DATASET_ROOT}/fer2013/{CSV_FILE}\n")
            sys.exit(1)

    model = EmotionRecognitionModel().to(device)

    if os.path.exists(SAVE_PATH):
        print(f"Loading existing model from '{SAVE_PATH}' to continue training...")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"No existing model found at '{SAVE_PATH}'. Starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Using device", device.type)

    train_loader = get_data_loader(train=True)
    val_loader = get_data_loader(train=False)

    print("\nStarting training...")

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        print()
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Get the predictions
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Summary:')
        print(
            f'Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%')

    print("Training finished!")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    _train_emotions_model()
