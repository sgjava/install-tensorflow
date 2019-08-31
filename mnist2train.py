"""
Copyright (c) Steven P. Goldsmith. All rights reserved.
"""

"""Train model with the MNIST database of handwritten digits. This example was taken from https://github.com/tankala/ai-examples and updated to TensorFlow 2.

@author: sgoldsmith

"""

import logging, numpy, tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

# Setup logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info("TensorFlow version %s" % tf.version)
# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshaping to format which CNN expects (batch, height, width, channels)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')
# Normalize inputs from 0-255 to 0-1
x_train /= 255
x_test /= 255
# One hot encode
number_of_classes = 10
y_train = utils.to_categorical(y_train, number_of_classes)
y_test = utils.to_categorical(y_test, number_of_classes)
# Create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax'))
# Compile model
logger.info("Compile model")
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# Train model
logger.info("Train model")
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=0)
# Save the entire model to a HDF5 file
logger.info("Save model")
model.save('output/mnist2.h5')
# Evaluate model
logger.info("Evaluate model")
accuracy_score = model.evaluate(x_test, y_test, verbose=0)
logger.info("%s: %4.2f, %s: %4.2f" % (model.metrics_names[0], accuracy_score[0], model.metrics_names[1], accuracy_score[1]))
