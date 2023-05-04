from datetime import datetime
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import h5py


print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


# Load dataset from an HDF5 file
def load_dataset(filename="2x2_dataset.h5"):
    with h5py.File(filename, "r") as f:
        images = f["images"][:]
        labels = f["labels"][:]
    return images, labels


# Load dataset
images, labels = load_dataset()
print(f"images.shape: {images.shape}")  # (num_images, 2, 2, 1)

# Preprocess the dataset
X = images
y = LabelEncoder().fit_transform(labels)  # Convert labels to integers

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the CNN architecture
# Input is 2x2 grayscale image
model = Sequential()
model.add(Conv2D(16, (2, 2), activation="relu", input_shape=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(7, activation="softmax"))

opt = Adam(learning_rate=(0.004 - 0.00036))

# Compile the model with the optimizer
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Print the model summary
model.summary()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)


# Train the model
model.fit(
    X_train,
    y_train,
    batch_size=32,  # Fixed
    epochs=150,
    validation_split=0.2,
    callbacks=[tensorboard_callback],
)


# Evaluate the model
_, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
model.save("2x2_cnn.h5")  # Save the model
