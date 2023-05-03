import os
import h5py
import numpy as np


def create_2x2_dataset(size=700):
    assert size % 7 == 0, "Size should be a multiple of 7 for a balanced dataset"
    num_samples_per_class = size // 7

    images = []
    labels = []

    # Black images
    for _ in range(num_samples_per_class):
        images.append(np.zeros((2, 2), dtype=np.uint8))
        labels.append("Black")

    # White images
    for _ in range(num_samples_per_class):
        images.append(np.full((2, 2), 255, dtype=np.uint8))
        labels.append("White")

    # Uniform images
    for _ in range(num_samples_per_class):
        img_value = np.random.randint(1, 255)
        img = np.full((2, 2), img_value, dtype=np.uint8)
        images.append(img)
        labels.append("Uniform")

    # Verical
    for _ in range(num_samples_per_class):
        img = np.zeros((2, 2), dtype=np.uint8)

        if np.random.choice([True, False]):
            img[0, 0] = img[1, 0] = np.random.randint(1, 255)
            img[0, 1] = np.random.randint(1, 255)
            while img[0, 1] == img[0, 0]:
                img[0, 1] = np.random.randint(1, 255)
            img[1, 1] = np.random.randint(1, 255)
            while img[1, 1] == img[0, 0] or img[1, 1] == img[0, 1]:
                img[1, 1] = np.random.randint(1, 255)
        else:
            img[0, 1] = img[1, 1] = np.random.randint(1, 255)
            img[0, 0] = np.random.randint(1, 255)
            while img[0, 0] == img[0, 1]:
                img[0, 0] = np.random.randint(1, 255)
            img[1, 0] = np.random.randint(1, 255)
            while img[1, 0] == img[0, 0] or img[1, 0] == img[0, 1]:
                img[1, 0] = np.random.randint(1, 255)

        images.append(img)
        labels.append("Vertical")

    # Horizontal
    for _ in range(num_samples_per_class):
        img = np.zeros((2, 2), dtype=np.uint8)

        if np.random.choice([True, False]):
            img[0, 0] = img[0, 1] = np.random.randint(1, 255)
            img[1, 0] = np.random.randint(1, 255)
            while img[1, 0] == img[0, 0]:
                img[1, 0] = np.random.randint(1, 255)
            img[1, 1] = np.random.randint(1, 255)
            while img[1, 1] == img[0, 0] or img[1, 1] == img[1, 0]:
                img[1, 1] = np.random.randint(1, 255)
        else:
            img[1, 0] = img[1, 1] = np.random.randint(1, 255)
            img[0, 0] = np.random.randint(1, 255)
            while img[0, 0] == img[1, 0]:
                img[0, 0] = np.random.randint(1, 255)
            img[0, 1] = np.random.randint(1, 255)
            while img[0, 1] == img[0, 0] or img[0, 1] == img[1, 0]:
                img[0, 1] = np.random.randint(1, 255)

        images.append(img)
        labels.append("Horizontal")

    # Diagonal
    for _ in range(num_samples_per_class):
        img = np.zeros((2, 2), dtype=np.uint8)

        img[0, 0] = img[1, 1] = np.random.randint(1, 255)
        img[0, 1] = np.random.randint(1, 255)
        while img[0, 1] == img[0, 0]:
            img[0, 1] = np.random.randint(1, 255)

        img[1, 0] = np.random.randint(1, 255)
        while img[1, 0] == img[0, 0] or img[1, 0] == img[0, 1]:
            img[1, 0] = np.random.randint(1, 255)

        images.append(img)
        labels.append("Diagonal")

    # Other images
    for _ in range(num_samples_per_class):
        unique_values = np.random.choice(np.arange(0, 256), size=4, replace=False)
        img = np.array(
            [
                [unique_values[0], unique_values[1]],
                [unique_values[2], unique_values[3]],
            ],
            dtype=np.uint8,
        )
        images.append(img)
        labels.append("Other")

    images = np.array(images)
    return images, labels


# Create dataset
images, labels = create_2x2_dataset()
images = images / 255.0
images = np.expand_dims(images, axis=-1)


# Save dataset to an HDF5 file
def save_dataset(images, labels, filename="2x2_dataset.h5"):
    filename = os.path.join(os.path.dirname(__file__), filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("labels", data=labels)


save_dataset(images, labels)
