import numpy as np

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Dataset:

# Loading the MNIST dataset
(train_img, train_lbl), (test_img, test_lbl) = mnist.load_data()

# Convert to float32
train_img = np.array(train_img, np.float32)
test_img = np.array(test_img, np.float32)

# Image normalization from [0,255] to [0,1]
train_img = train_img / 255.0
test_img = test_img / 255.0

# Function to plot train history
def show_train_history(train_history, train_metric, val_metric):
  plt.figure(figsize=(10,6))
  plt.plot(train_history.history[train_metric], label=f'Training {train_metric}'),
  plt.plot(train_history.history[val_metric], label=f'Validation {val_metric}'),
  plt.title('Training and Validation ' + train_metric)
  plt.ylabel(train_metric)
  plt.xlabel('Epochs')
  plt.legend(['Training', 'Validation'], loc='upper left')
  plt.grid(True)

# Artifcial Neural Network (ANN) (30 Points):

# Building the ANN model
ann_model = Sequential([
          Flatten(input_shape=(28, 28)),
          Dense(256, activation='relu'),
          Dense(128, activation='relu'),
          Dense(10, activation='softmax')
        ])

# Compiling the model
ann_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

# Model training with validation split 10%
ann_history = ann_model.fit(train_img, train_lbl, epochs=10, validation_split=0.1, batch_size=32)
print()

# Plot accuracy and loss for ANN
show_train_history(ann_history, 'accuracy', 'val_accuracy')
show_train_history(ann_history, 'loss', 'val_loss')
print()

# Evaluating on test data
test_loss, test_acc = ann_model.evaluate(test_img, test_lbl)
print(f'Test Accuracy: {test_acc}')
print()

# Predicting on test data
pred = ann_model.predict(test_img)
pred_lbl = np.argmax(pred, axis=1)
print()

# Computing confusion matrix
comf_matrix = confusion_matrix(test_lbl, pred_lbl)
print('Confusion Matrix:')
print(comf_matrix)
print()

# Convolutional Neural Network (CNN) (30 Points):

# Reshape data to fit model
X_train = train_img.reshape(train_img.shape[0], 28, 28, 1)
X_test = test_img.reshape(test_img.shape[0], 28, 28, 1)

# Create CNN model
cnn_model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

# Compile the model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
cnn_history = cnn_model.fit(train_img, train_lbl, validation_split=0.1, epochs=10, batch_size=32)
print()

# Plot accuracy and loss for CNN
show_train_history(cnn_history, 'accuracy', 'val_accuracy')
show_train_history(cnn_history, 'loss', 'val_loss')
print()

# Evaluating on test data
test_loss, test_acc = cnn_model.evaluate(test_img, test_lbl)
print(f'Test Accuracy: {test_acc}')
print()

# Predicting on test data
pred = cnn_model.predict(test_img)
pred_lbl = np.argmax(pred, axis=1)
print()

# Computing confusion matrix
comf_matrix = confusion_matrix(test_lbl, pred_lbl)
print('Confusion Matrix:')
print(comf_matrix)
print()
