'''
Authors: Anthony Badea
Date: May 8, 2022
'''

# python imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import tensorflow as tf
import os
import sys
import logging

# custom code
from model import get_particle_net, get_particle_net_lite
from get_data import get_data

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(config = None):

    ops = options()

    # logger
    logging.basicConfig(level = 'INFO', format = '%(levelname)s: %(message)s')
    log = logging.getLogger()

    # protection
    if ops.model_weights is None:
        log.error('ERROR: no model weights were provided, exiting')
        sys.exit(1)

    # load data
    if ops.background:
        x, weights = get_data({"inFileName" : ops.inFile, "background" : ops.background})
    else:
        x_train, y_train, weights_train, x_test, y_test, weights_test = get_data({"inFileName" : ops.inFile, "background" : ops.background})
        x = {key : np.concatenate([x_train[key], x_test[key]]) for key in x_train.keys()}
        y = np.concatenate([y_train, y_test])
        weights = np.concatenate([weights_train, weights_test])

    # load model
    num_classes = 3
    input_shapes = {k:x[k].shape[1:] for k in x}
    model = get_particle_net_lite(num_classes, input_shapes)

    # if checkpoint directory provided use the latest
    if os.path.isdir(ops.model_weights):
        latest = tf.train.latest_checkpoint(ops.model_weights)
        log.info(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    else:
        model.load_weights(ops.model_weights)
    
    # make model prediction
    p = model.predict(x)

    # compute mass
    pt = x["features"][:,:,0]
    eta = x["features"][:,:,1]
    phi = x["features"][:,:,2]
    m = x["features"][:,:,3]

    # convert to cartesian
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(m**2 + px**2 + py**2 + pz**2)
    jets = np.stack([e,px,py,pz],axis=-1)

    # prepare output file
    outData = {}

    # compute parents four mom and masses
    if not ops.background:
        y_parents = np.einsum("bjk, bkn -> bjn", np.transpose(y,[0,2,1]), jets)
        y_masses = np.sqrt(y_parents[:,:,0]**2 - y_parents[:,:,1]**2 - y_parents[:,:,2]**2 - y_parents[:,:,3]**2)
        plt.hist(y_masses[:,:2].mean(1),bins=np.linspace(0,2500,50), histtype="step", color="black", label="truth", weights=weights, density=True)
        plt.hist(y_masses[:,2].flatten(),bins=np.linspace(0,2500,50), alpha=0.5, color="black", label="truth isr", weights=weights, density=True)
        outData["mass_true"] = y_masses[:,:2].flatten()
        outData["weights_true"] = weights.reshape(-1,1).repeat(2,1).flatten()

    # compute parents four mom and masses
    p_parents = np.einsum("bjk, bkn -> bjn", np.transpose(p,[0,2,1]), jets)
    p_masses = np.sqrt(p_parents[:,:,0]**2 - p_parents[:,:,1]**2 - p_parents[:,:,2]**2 - p_parents[:,:,3]**2)
    plt.hist(p_masses[:,:2].mean(1),bins=np.linspace(0,2500,50), histtype="step", color="red", label="prediction", weights=weights, density=True)
    plt.hist(p_masses[:,2].flatten(),bins=np.linspace(0,2500,50), alpha=0.5, color="red", label="prediction isr", weights=weights, density=True)
    outData["mass_pred"] = p_masses[:,:2].flatten()
    outData["weights_pred"] = weights.reshape(-1,1).repeat(2,1).flatten()

    # save to file
    for key, val in outData.items():
        print(key,val.shape)
    np.savez("out.npz", **outData)

    # plot
    plt.xlabel("m_{avg} [GeV]")
    plt.ylabel("Density of Events")
    plt.legend()
    plt.savefig("eval.pdf")
    # plt.show()

    

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    parser.add_argument("-b",  "--background", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()



