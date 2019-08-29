"""
Copyright (c) Steven P. Goldsmith. All rights reserved.
"""

"""Load the model trained with the MNIST database of handwritten digits.

@author: sgoldsmith

"""

import logging, tensorflow as tf

# Setup logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info("TensorFlow version %s" % tf.version)
# Recreate the exact same model, including its weights and the optimizer
logger.info("Load model")
model = tf.keras.models.load_model('output/mnist.h5')
# Show the model architecture
model.summary()
