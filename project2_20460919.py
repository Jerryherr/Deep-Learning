# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:04:25 2017

@author: Jerryho
"""

import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing the directory of an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    f = open(data_dir)
    paths = f.readlines()
    f.close()

    labels = []
    images = []
    for line in paths:
        im, lab = line.split(' ')
        images.append(im)
        labels.append(int(lab.rstrip()))

    return images, labels


# Load training and testing datasets.
TRAIN_PATH = '/train.txt'
VAL_PATH = '/val.txt'
TEST_PATH = '/test.txt'
work_dir = os.getcwd()
train_data_dir = os.path.join(work_dir, TRAIN_PATH)
val_data_dir = os.path.join(work_dir, VAL_PATH)
test_data_dir = os.path.join(work_dir, TEST_PATH)

images, labels = load_data(train_data_dir)

# Read the photos
Images = []
for i in images:
    Images.append(skimage.data.imread(i))


# Resize images
images64 = [skimage.transform.resize(image, (64, 64)) for image in Images]
display_images_and_labels(images64, labels)

    
labels_new = np.array(labels)
images_new = np.array(images64)
print("labels: ", labels_new.shape, "\nimages: ", images_new.shape)

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer. 
    logits = tf.contrib.layers.fully_connected(images_flat, 30, tf.nn.relu)

    predicted_labels = tf.argmax(logits, 1)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()


# Create a session to run the graph we created.
session = tf.Session(graph=graph)

temp = session.run([init])

# Training process
for i in range(500):
    temp, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_new, labels_ph: labels_new})
    if i % 20 == 0:
        print("Loss: ", loss_value)

# Validation
val, label_val = load_data(val_data_dir)

image_val = []
for v in val:
    image_val.append(skimage.data.imread(v))
display_images_and_labels(image_val, label_val)

# Resize val images
image_val64 = [skimage.transform.resize(image, (64, 64)) for image in image_val]
#display_images_and_labels(image_val64, label_val)

predicted = session.run([predicted_labels], feed_dict={images_ph: image_val64})[0]
print(predicted)

# Calculate the accuracy
match_count = sum([int(y == y_) for y, y_ in zip(label_val, predicted)])
accuracy = match_count / len(label_val)
print("Accuracy: {:.3f}".format(accuracy))


# =============================================================================
# # Load the test dataset.
# test_images, test_labels = load_data(test_data_dir)
# 
# # Transform the images, just like we did with the training set.
# test_images64 = [skimage.transform.resize(image, (64, 64))
#                  for image in test_images]
# display_images_and_labels(test_images64, test_labels)
# 
# # Run predictions against the full test set.
# predicted_test = session.run([predicted_labels], 
#                         feed_dict={images_ph: test_images64})[0]
# =============================================================================
