"""
Copyright (c) Steven P. Goldsmith. All rights reserved.
"""

"""Predict sample digit images using MNIST model 1.

@author: sgoldsmith

"""
import logging, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Setup logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info("TensorFlow version %s" % tf.version)
model = load_model('output/mnist1.h5')
for index in range(10):
    img = Image.open('images/' + str(index) + '.png').convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28).astype(float)
    # Predicting the Test set results
    y_pred = model.predict(im2arr)
    if y_pred[0, index] == 1:
        logger.info("%d predicted     %s" % (index, y_pred))
    else:
        logger.info("%d not predicted %s" % (index, y_pred))
