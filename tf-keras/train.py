'''
The ParticleNet model can be obtained by calling the get_particle_net function in particle_net.py, which can return either an MXNet Symbol or an MXNet Gluon HybridBlock. The model takes three input arrays:

points: the coordinates of the particles in the (eta, phi) space. It should be an array with a shape of (N, 2, P), where N is the batch size and P is the number of particles.
features: the features of the particles. It should be an array with a shape of (N, C, P), where N is the batch size, C is the number of features, and P is the number of particles.
mask: a mask array with a shape of (N, 1, P), taking a value of 0 for padded positions.
'''


############################
#   IMPORTS & SETTINGS     #
############################

# general python
import os
import math
import numpy as np

# logging
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger()

# Tensorflow GPU settings
import tensorflow as tf
from tensorflow import keras
print(f"Tensorflow version {tf.__version__}")
print(f"Tensorflow executing in eager mode: {tf.executing_eagerly()}")
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPUs Available, using CPU")

# argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inFile", help="Input file", required=True, type=str)
ops = parser.parse_args()

# custom code
from model import get_particle_net, get_particle_net_lite
from get_data import get_data

############################
#      LOAD DATA           #
############################

x_train, y_train, x_test, y_test = get_data({"inFileName" : ops.inFile})


############################
#     MAKE MODEL           #
############################
model_type = 'particle_net_lite'
num_classes = y_train.shape[1]
input_shapes = {k:x_train[k].shape[1:] for k in x_train}
print(f"Number of classes {num_classes} and graph shapes {input_shapes}")

if 'lite' in model_type:
    model = get_particle_net_lite(num_classes, input_shapes)
else:
    model = get_particle_net(num_classes, input_shapes)

# now need to update to unsupervised loss
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=10e-3),
              metrics=['accuracy'])
model.run_eagerly = True
# model.summary()

# Prepare model model saving directory.
save_dir = 'model_checkpoints'
model_name = '%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint]


############################
#     DO TRAINING          #
############################
batch_size = 256
epochs = 30
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks,
          verbose=1)