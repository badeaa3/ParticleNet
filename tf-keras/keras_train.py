'''
The ParticleNet model can be obtained by calling the get_particle_net function in particle_net.py, which can return either an MXNet Symbol or an MXNet Gluon HybridBlock. The model takes three input arrays:

points: the coordinates of the particles in the (eta, phi) space. It should be an array with a shape of (N, 2, P), where N is the batch size and P is the number of particles.
features: the features of the particles. It should be an array with a shape of (N, C, P), where N is the batch size, C is the number of features, and P is the number of particles.
mask: a mask array with a shape of (N, 1, P), taking a value of 0 for padded positions.
'''


############################
#   IMPORTS & SETTINGS     #
############################
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite, get_particle_net_evt
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPUs Available, using CPU")



############################
#      LOAD DATA           #
############################
f = np.load("signal_UDB_UDS_training_v33_cannonball.npz",allow_pickle=True)
labels = f["labels"]
event_selection = (labels[:,0] != -1)
jets = f["jets"][event_selection]
labels = labels[event_selection]
e = jets[:,:,0]
pt  = np.sqrt(jets[:,:,1]**2 + jets[:,:,2]**2)
eta = np.nan_to_num(np.arcsinh(jets[:,:,3]/pt),0)
phi = np.arctan2(jets[:,:,2],jets[:,:,1])
# make input formats (NOTE: difference from the original readme the channel needs to go last so the order is (N,P,C))
points = np.stack([eta,phi],-1)
features = np.stack([pt,e],-1)
mask = np.expand_dims((jets == 0).sum(-1) == 0,-1)

# get indices of isr, gluino 1 and 2
idx = np.repeat(np.arange(jets.shape[1]).reshape(1,-1),jets.shape[0],0)
issig = np.repeat(labels[:,:-5],1,-1).astype(bool)
isr = idx[~issig].reshape(jets.shape[0],2)
glu = idx[issig].reshape(jets.shape[0],6)
match = np.concatenate([np.ones((labels.shape[0],1)),labels[:,-5:]],axis=1).astype(bool)
glu1 = glu[match].reshape(jets.shape[0],3)
glu2 = glu[~match].reshape(jets.shape[0],3)
# convert to one-hot format
ISR = np.zeros((jets.shape[0],jets.shape[1]))
np.put_along_axis(ISR,isr,1,axis=1)
GL1 = np.zeros((jets.shape[0],jets.shape[1]))
np.put_along_axis(GL1,glu1,1,axis=1)
GL2 = np.zeros((jets.shape[0],jets.shape[1]))
np.put_along_axis(GL2,glu2,1,axis=1)
# stack to multiclass label
labels = np.stack([GL1,GL2,ISR],axis=-1)
labels = np.ones((jets.shape[0],3))

# do train test split
split = int(0.8 * jets.shape[0])
train_x = {
    "points": points[:split],
    "features" : features[:split],
    "mask" : mask[:split]
}
train_y = labels[:split]
val_x = {
    "points": points[split:],
    "features" : features[split:],
    "mask" : mask[split:]
}
val_y = labels[split:]

# shuffle data
def shuffle(values, label, seed=None):
    if seed is not None:
        np.random.seed(seed)
    shuffle_indices = np.arange(len(label))
    np.random.shuffle(shuffle_indices)
    for k in values:
        values[k] = values[k][shuffle_indices]
    label = label[shuffle_indices]
    return values, label

train_x, train_y = shuffle(train_x, train_y)



############################
#     MAKE MODEL           #
############################
model_type = 'particle_net_evt' # choose between 'particle_net' and 'particle_net_lite'
num_classes = train_y.shape[1]
input_shapes = {k:train_x[k].shape[1:] for k in train_x}
print(num_classes,input_shapes, train_y.shape)
if 'lite' in model_type:
    model = get_particle_net_lite(num_classes, input_shapes)
elif 'evt' in model_type:
    model = get_particle_net_evt(num_classes, input_shapes)
else:
    model = get_particle_net(num_classes, input_shapes)

# learning rate
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    logging.info('Learning rate: %f'%lr)
    return lr

def custom_loss(y_true, y_pred):
    # do things with y_pred
    print(y_pred)
    masses = y_pred[:,:,0]
    glu = masses[:,:2]
    mu = 1400 #tf.reduce_mean(glu)
    std = 30 #tf.math.reduce_std(glu)
    n = -1 * tf.exp(-0.5 * ((glu - mu)/std)**2) / (std * (2*math.pi)**0.5)
    return tf.reduce_sum(n)

# now need to update to unsupervised loss
model.compile(loss=custom_loss, #'mean_squared_error', #'binary_crossentropy', #'categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
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
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()
callbacks = [checkpoint, lr_scheduler, progress_bar]



############################
#     DO TRAINING          #
############################
batch_size = 100 if 'lite' in model_type else 384
epochs = 1
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(val_x, val_y),
          shuffle=True,
          callbacks=callbacks,
          verbose=2)