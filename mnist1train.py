"""
Copyright (c) Steven P. Goldsmith. All rights reserved.
"""

"""Train model with the MNIST database of handwritten digits. This example was taken from the official TensorFlow site.

@author: sgoldsmith

"""

import logging, tensorflow as tf

# Setup logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info("TensorFlow version %s" % tf.version)
# The MNIST database of handwritten digits,, has a training set of 60,000 examples, and a test set of 10,000 examples.
mnist = tf.keras.datasets.mnist
# Split into test and train pairs
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Define a simple sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
# Compile model
logger.info("Compile model")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train model
logger.info("Train model")
model.fit(x_train, y_train, epochs=5, verbose=0)
# Save the entire model to a HDF5 file
logger.info("Save model")
model.save('output/mnist1.h5')
# Evaluate model
logger.info("Evaluate model")
accuracy_score = model.evaluate(x_test, y_test, verbose=0)
logger.info("%s: %4.2f, %s: %4.2f" % (model.metrics_names[0], accuracy_score[0], model.metrics_names[1], accuracy_score[1]))
