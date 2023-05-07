import os
import h5py
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_dataset(filename="2x2_dataset.h5"):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with h5py.File(file_path, "r") as f:
        images = f["images"][:]
        labels = [label.decode("utf-8") for label in f["labels"][:]]
    return images, labels


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (2, 2))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(model, images, labels, num_epochs=15000, learning_rate=1e-3):
    wandb.init(
        mode="offline",
        project="2x2-nn",
        config={"num_epochs": num_epochs, "learning_rate": learning_rate},
    )
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels_encoded.shape}")
    print("Unique labels:", np.unique(labels_encoded))
    print("Number of classes:", len(np.unique(labels_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    X_train = torch.from_numpy(X_train).transpose(1, 3).to(device)
    X_test = torch.from_numpy(X_test).transpose(1, 3).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_train).sum().item()
            accuracy = correct / y_train.size(0) * 100
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
            )
            wandb.log({"Loss": loss.item(), "Accuracy": accuracy})

    return model
