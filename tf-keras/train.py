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
parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=10e-3, type=float)
parser.add_argument("-b", "--batch_size", help="Batch size", default=256, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=1, type=int)
ops = parser.parse_args()

# custom code
from model import get_particle_net, get_particle_net_lite
import get_data

############################
#      LOAD DATA           #
############################
return_graph = False
if return_graph:
    x_train, y_train, weights_train, x_test, y_test, weights_test = get_data.get_data(inFileName = ops.inFile)
    loss = 'categorical_crossentropy'
else:
    x_train, y_train, weights_train, x_test, y_test, weights_test = get_data.get_signal_and_background(signal="signal_1500_UDB_UDS_training_v65.h5", background="user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5")
    loss = {
        "nodes" : 'categorical_crossentropy', # nodes
        "graph" : 'binary_crossentropy' # graph
    }

############################
#     MAKE MODEL           #
############################
model_type = 'particle_net_lite'
num_classes = 1 #y_train.shape[1]
input_shapes = {k:x_train[k].shape[1:] for k in x_train}
print(f"Number of classes {num_classes} and graph shapes {input_shapes}")
model = get_particle_net_lite(num_classes, input_shapes, return_graph)

def loss_fn(y_true, y_pred):

    # unpack inputs
    nodes_true = y_true[:,:-1]
    nodes_true = tf.reshape(nodes_true, (-1,8,3))
    graph_true = y_true[:,-1]
    nodes_pred = tf.reshape(y_pred[:,:-1], (-1,8,3))
    graph_pred = y_pred[:,-1]

    # compute loss
    graph = tf.keras.losses.BinaryCrossentropy(from_logits=True)(graph_true, graph_pred)
    nodes = 2*tf.keras.losses.CategoricalCrossentropy()(nodes_true, nodes_pred,sample_weight=graph_true)
    loss = graph + nodes
    return loss

# now need to update to unsupervised loss
model.compile(loss=loss_fn,
              optimizer=keras.optimizers.Adam(learning_rate=ops.learning_rate),
              metrics=['accuracy'])
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
model.fit(x_train, y_train,
          batch_size=ops.batch_size,
          epochs=ops.epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks,
          verbose=1)