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
parser.add_argument("-is", "--inSigFile", help="Input file", type=str)
parser.add_argument("-ib", "--inBkgFile", help="Input file", type=str)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=10e-3, type=float)
parser.add_argument("-b", "--batch_size", help="Batch size", default=256, type=int)
parser.add_argument("-n", "--num_batches", help="Number of batches per epoch", default=1, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs", default=1, type=int)
ops = parser.parse_args()

# custom code
from model import get_particle_net, get_particle_net_lite
from WeightedSamplingDataLoader import WeightedSamplingDataLoader, loadWeightSamples, formInput

############################
#      LOAD DATA           #
############################

njets = 8
signal = ops.inSigFile
background = ops.inBkgFile
fileList = sorted([line.strip() for line in open(ops.inBkgFile,"r")])
load = False
probabilities, fileidx, usedFiles = loadWeightSamples(load, "WeightSamplerDijets.npz" if load else fileList)
train_dataset = WeightedSamplingDataLoader(njets, signal, probabilities, fileidx, fileList if load else usedFiles, ops.num_batches, ops.batch_size).map(formInput).prefetch(tf.data.AUTOTUNE)
validation_dataset = WeightedSamplingDataLoader(njets, signal, probabilities, fileidx, fileList if load else usedFiles, ops.num_batches, ops.batch_size).map(formInput).prefetch(tf.data.AUTOTUNE)

############################
#     MAKE MODEL           #
############################
model_type = 'particle_net_lite'
model = get_particle_net_lite(num_classes = 1, input_shapes = {'points': (8, 2), 'features': (8, 4), 'mask': (8, 1)}, return_graph = False)

def loss_fn(y_true, y_pred):

    # unpack inputs
    nodes_true = y_true[:,:-1]
    nodes_true = tf.reshape(nodes_true, (-1,8,3))
    graph_true = y_true[:,-1]
    nodes_pred = tf.reshape(y_pred[:,:-1], (-1,8,3))
    graph_pred = y_pred[:,-1]

    # compute loss
    graph = tf.keras.losses.BinaryCrossentropy(from_logits=True)(graph_true, graph_pred)
    nodes = tf.keras.losses.CategoricalCrossentropy()(nodes_true, nodes_pred,sample_weight=graph_true)
    loss = 0*graph + nodes
    return loss

# now need to update to unsupervised loss
model.compile(loss=loss_fn,
              optimizer=keras.optimizers.Adam(learning_rate=ops.learning_rate)) #,
              # metrics=['accuracy'])
# model.summary()

# Prepare model saving directory.
save_dir = 'model_checkpoints'
model_name = '%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=10, mode="min", restore_best_weights=True, monitor="val_loss"), 
    tf.keras.callbacks.TerminateOnNaN()
]


############################
#     DO TRAINING          #
############################
model.fit(train_dataset,
          batch_size=ops.batch_size,
          epochs=ops.epochs,
          validation_data=validation_dataset,
          shuffle=True,
          callbacks=callbacks,
          verbose=1)