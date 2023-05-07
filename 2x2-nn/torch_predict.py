import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_model import CNN


def load_dataset(filename="2x2_dataset.h5"):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with h5py.File(file_path, "r") as f:
        images = f["images"][:]
        labels = [label.decode("utf-8") for label in f["labels"][:]]
    return images, labels


def predict_label(sample_image, loaded_model, label_encoder):
    sample_image = np.expand_dims(sample_image, axis=0)
    sample_image = torch.from_numpy(sample_image).transpose(1, 3).to(device)
    with torch.no_grad():
        predicted_probs = loaded_model(sample_image)
    predicted_class = torch.argmax(predicted_probs)
    predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
    return predicted_label


def display_image_and_labels(images, true_label, predicted_label, index):
    plt.imshow(images[index].reshape(2, 2), cmap="gray", vmin=0, vmax=1)
    plt.title(f"True label: {true_label} | Predicted label: {predicted_label}")
    plt.axis("off")
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

images, labels = load_dataset()
label_encoder = LabelEncoder().fit(labels)
loaded_model = CNN().to(device)
loaded_model.load_state_dict(torch.load("./output/torch_trained.pth"))
loaded_model.eval()

index = 0

while True:
    print(f"Index: {index}")
    true_label = labels[index]
    predicted_label = predict_label(images[index], loaded_model, label_encoder)

    display_image_and_labels(images, true_label, predicted_label, index)

    user_input = input("Press 'q' to continue or any other key to quit: ")
    if user_input.lower() != "q":
        break
    else:
        index = np.random.randint(0, len(images))
        if index >= len(images):
            print("You have reached the end of the dataset.")
            break
