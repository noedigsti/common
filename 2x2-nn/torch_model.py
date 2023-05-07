from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import h5py


def label_encoder(labels):
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_index[label] for label in labels]
    return encoded_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Load dataset from an HDF5 file
def load_dataset(filename="2x2_dataset.h5"):
    with h5py.File(filename, "r") as f:
        images = f["images"][:]
        labels = f["labels"][:]
    return images, labels


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# Load dataset
images, labels = load_dataset()
print(f"images.shape: {images.shape}")  # (num_images, 2, 2, 1)

# Preprocess the dataset
X = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
y = torch.tensor(label_encoder(labels), dtype=torch.long).to(device)

# Create a custom dataset
dataset = CustomDataset(X, y)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the CNN architecture
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
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = CNN().to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.004 - 0.00036)
criterion = nn.CrossEntropyLoss()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_writer = SummaryWriter(log_dir=logdir)


# Train the model
num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # Calculate the accuracy for the current epoch
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
    )

    # Log the loss and accuracy for the current epoch
    tensorboard_writer.add_scalar("Loss", loss.item(), epoch)
    tensorboard_writer.add_scalar("Accuracy", accuracy, epoch)

# Save the model
torch.save(model.state_dict(), "2x2_cnn.pth")
