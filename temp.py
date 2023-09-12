# # 🧠 Alzheimer's Disease Classification
# 🕸️ A Convolutional Neural Network (CNN) model is used here to classify brain MRIs into normal, very-mild, mild and moderate Alzheimer classes. The data in total consists of 6400 images.
# Developed as part of a project work for the **UCS 1603 Introduction to Machine Learning** Course. 📖
# Authors:
# * Shashanka Venkatesh  - 18 5001 145
# * Suraj Jain           - 18 5001 177
# * Vishakan Subramanian - 18 5001 196
# * Vishnu Krishnan      - 18 5001 200
# **We recommend the use of a GPU Accelerator to reduce the load on the CPU and to run the notebook faster.**
# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree

from PIL import Image
from random import randint

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from keras import Sequential, Input
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D

from keras.optimizers import RMSprop
from keras.losses import BinaryCrossentropy


print("TensorFlow Version:", tf.__version__)
# Data Pre-Processing
base_dir = "./kaggle_dataset/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

if os.path.exists(work_dir):
    remove_tree(work_dir)


os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))
WORK_DIR = './dataset/'

CLASSES = ['NonDemented',
           'Demented'
           ]

IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)
# Performing Image Augmentation to have more data samples

ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(rescale=1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM,
              data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

train_data_gen = work_dr.flow_from_directory(
    directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)


def show_images(generator, y_pred=None):
    """
    Input: An image generator,predicted labels (optional)
    Output: Displays a grid of 9 images with lables
    """

    # get image lables
    labels = dict(zip([0, 1], CLASSES))

    # get a batch of images
    x, y = generator.next()

    # display a grid of 9 images
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            idx = randint(0, 6400)
            plt.imshow(x[idx])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[idx])]))

    else:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(
                labels[np.argmax(y[i])], labels[y_pred[i]]))


# Display Train Images
show_images(train_data_gen)
# Retrieving the data from the ImageDataGenerator iterator

train_data, train_labels = train_data_gen.next()
# Getting to know the dimensions of our dataset

print(train_data.shape, train_labels.shape)
# Performing over-sampling of the data, since the classes are imbalanced

sm = SMOTE(random_state=42)

train_data, train_labels = sm.fit_resample(
    train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print(train_data.shape, train_labels.shape)
# Splitting the data into train, test, and validation sets

train_data, test_data, train_labels, test_labels = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42)
# Using the InceptionV3 model as a base model for the task
inception_model = InceptionV3(input_shape=(
    176, 176, 3), include_top=False, weights="imagenet")
for layer in inception_model.layers:
    layer.trainable = False
custom_inception_model = Sequential([
    inception_model,
    Dropout(0.5),
    GlobalAveragePooling2D(),
    Flatten(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
], name="inception_cnn_model")
# Defining a custom callback function to stop training our model when accuracy goes above 99%


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True


my_callback = MyCallback()

# ReduceLROnPlateau to stabilize the training process of the model
rop_callback = ReduceLROnPlateau(monitor="val_loss", patience=3)
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'),
           tfa.metrics.F1Score(num_classes=1)]

CALLBACKS = [my_callback, rop_callback]

custom_inception_model.compile(optimizer=RMSprop(),
                               loss=BinaryCrossentropy(),
                               metrics=METRICS)

custom_inception_model.summary()
# Fit the training data to the model and validate it using the validation data
EPOCHS = 20

history = custom_inception_model.fit(train_data, train_labels, validation_data=(
    val_data, val_labels), callbacks=CALLBACKS, epochs=EPOCHS)
# Tabulating the Results of our custom InceptionV3 model
# Plotting the trend of the metrics during training

fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
# Evaluating the model on the data

# train_scores = model.evaluate(train_data, train_labels)
# val_scores = model.evaluate(val_data, val_labels)
test_scores = custom_inception_model.evaluate(test_data, test_labels)

# print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
# print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))
# Predicting the test data

pred_labels = custom_inception_model.predict(test_data)
# Print the classification report of the tested data

# Since the labels are softmax arrays, we need to roundoff to have it in the form of 0s and 1s,
# similar to the test_labels


def roundoff(arr):
    """To round off according to the argmax of each predicted label array. """
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr


for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))
# Plot the confusion matrix to understand the classification in detail

pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(test_ls, pred_ls)

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=CLASSES,
                 yticklabels=CLASSES)

plt.title('Alzheimer\'s Disease Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show(ax)
# Printing some other classification metrics

print("Balanced Accuracy Score: {} %".format(
    round(BAS(test_ls, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(
    round(MCC(test_ls, pred_ls) * 100, 2)))
# Saving the model for future use

custom_inception_model_dir = work_dir + "alzheimer_inception_cnn_model"
custom_inception_model.save(custom_inception_model_dir, save_format='h5')
os.listdir(work_dir)
pretrained_model = tf.keras.models.load_model(custom_inception_model_dir)

# Check its architecture
plot_model(pretrained_model, to_file=work_dir + "model_plot.png",
           show_shapes=True, show_layer_names=True)
# Using a custom CNN model for the task
