import os
import h5py
import matplotlib.pyplot as plt


def load_dataset(filename="2x2_dataset.h5"):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with h5py.File(file_path, "r") as f:
        images = f["images"][:]
        labels = [label.decode("utf-8") for label in f["labels"][:]]
    return images, labels


def show_image(images, index):
    plt.imshow(images[index].reshape(2, 2), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.show()


images, labels = load_dataset()
index = 0

label_map = {
    "1": "Black",
    "2": "White",
    "3": "Uniform",
    "4": "Horizontal",
    "5": "Vertical",
    "6": "Diagonal",
    "7": "Other",
}

while True:
    show_image(images, index)
    print("Image label:", labels[index])
    choice = input("Enter 1, 2, 3, 4, 5, 6, 7: ")

    if choice in label_map:
        target_label = label_map[choice]
        found = False
        for i in range(index + 1, len(labels)):
            if labels[i] == target_label:
                index = i
                found = True
                break

        if not found:
            for i in range(index):
                if labels[i] == target_label:
                    index = i
                    found = True
                    break

        if not found:
            print(f"No {target_label} label found in the dataset.")
    else:
        break
