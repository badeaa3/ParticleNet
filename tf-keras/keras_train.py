'''
The ParticleNet model can be obtained by calling the get_particle_net function in particle_net.py, which can return either an MXNet Symbol or an MXNet Gluon HybridBlock. The model takes three input arrays:

points: the coordinates of the particles in the (eta, phi) space. It should be an array with a shape of (N, 2, P), where N is the batch size and P is the number of particles.
features: the features of the particles. It should be an array with a shape of (N, C, P), where N is the batch size, C is the number of features, and P is the number of particles.
mask: a mask array with a shape of (N, 1, P), taking a value of 0 for padded positions.
'''

# python imports
import h5py
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import gc
import datetime
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

# custom code
from tf_keras_model import get_particle_net_evt

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():

    # user options
    ops = options()

    # set seeds to get reproducible results (only if requested)
    if ops.seed is not None:
        try:
            python_random.seed(ops.seed)
            np.random.seed(ops.seed)
            tf.random.set_seed(ops.seed)
        except:  # deprecated in newer tf versions
            tf.keras.utils.set_random_seed(ops.seed)

    # load data
    with h5py.File(ops.inFile,"r") as file:
        # points (NOTE: difference from the original readme the channel needs to go last so the order is (N,P,C))
        p = np.stack([file['source']['eta'],file['source']['phi']],-1)
        # features
        f = np.stack([file['source']['mass'],file['source']['pt']],-1)
        # reco jet -> gluino assignments
        g1 = np.stack([file['g1'][f'q{i}'] for i in range(1,4)],-1)
        g2 = np.stack([file['g2'][f'q{i}'] for i in range(1,4)],-1)

    # mask for padded jets (NOTE: difference from the original readme the channel needs to go last so the order is (N,P,C))
    mask = np.expand_dims((f[:,:,0] != 0),-1)

    # labels for gluinos
    y_g1 = np.zeros((p.shape[0],p.shape[1]+1))
    np.put_along_axis(y_g1, g1, 1, 1)
    y_g2 = np.zeros((p.shape[0],p.shape[1]+1))
    np.put_along_axis(y_g2, g2, 1, 1)
    y = np.stack([y_g1,y_g2],-1)
    # label for isr
    isr = np.expand_dims((y.sum(-1) == 0),-1)
    # concatenate
    y = np.concatenate([y,isr],-1)
    # remove extra row used for handling gluino index of -1
    y =  y[:,:p.shape[1]] 
    del g1, g2, isr
    gc.collect()

    # split samples
    p_train, p_test, f_train, f_test, mask_train, mask_test, y_train, y_test = train_test_split(p, f, mask, y, test_size=0.75, shuffle=True)
    x_train = {"points": p_train, "features": f_train, "mask": mask_train}
    x_test = {"points": p_test, "features": f_test, "mask": mask_test}
    del p, f, mask, y
    gc.collect()

    # make model
    num_classes = y_train.shape[1]
    input_shapes = {key:val.shape[1:] for key, val in x_train.items()}
    print(f"Num classes {num_classes}, Input shapes {input_shapes}, Train target shape {y_train.shape}, Test target shape {y_test.shape}")
    model = get_particle_net_evt(num_classes, input_shapes)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=ops.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # make callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=30, mode="min", restore_best_weights=True)) #, monitor="val_loss"))
    # ModelCheckpoint
    checkpoint_filepath = f'./checkpoints/training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}/' + "cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=False, save_weights_only=True,))
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    # callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

    # train
    history = model.fit(
        x_train, y_train,
        batch_size=ops.batch_size,
        epochs=ops.epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(x_test,y_test)
    )

    # plot loss
    plt.figure('loss_vs_epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.yscale('log'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(ops.outDir,'loss_vs_epochs.pdf'))

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", default="./")
    parser.add_argument("-e", "--epochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-b", "--batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
