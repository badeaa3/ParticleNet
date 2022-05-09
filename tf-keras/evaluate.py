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
import get_data

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

    # # load data
    # if ops.background:
    #     x, weights = get_data.get_data(inFileName = ops.inFile, background = ops.background)
    # else:
    #     # x_train, y_train, weights_train, x_test, y_test, weights_test = get_data.get_data(inFileName = ops.inFile)
    #     x_train, y_train, weights_train, x_test, y_test, weights_test = get_data.get_signal_and_background(signal="signal_1500_UDB_UDS_training_v65.h5", background="user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5")
    #     x = {key : np.concatenate([x_train[key], x_test[key]]) for key in x_train.keys()}
    #     y = np.concatenate([y_train, y_test])
    #     y_nodes = tf.reshape(y[:,:-1], (-1,8,3))
    #     y_graph = y[:,-1]
    #     weights = np.concatenate([weights_train, weights_test])

    # load signal
    s_x_train, s_y_train, s_weights_train, s_x_test, s_y_test, s_weights_test = get_data.get_data("signal_1500_UDB_UDS_training_v65.h5")
    points = np.concatenate([s_x_train["points"], s_x_test["points"]])
    features = np.concatenate([s_x_train["features"], s_x_test["features"]])
    mask = np.concatenate([s_x_train["mask"], s_x_test["mask"]])
    # combine signal targets
    s_y_nodes = np.concatenate([s_y_train, s_y_test])
    s_y_graph = np.ones((s_y_nodes.shape[0],1))
    s_weights = np.concatenate([s_weights_train, s_weights_test])
    s_x = {
        "points" : points,
        "features" : features,
        "mask" : mask
    }

    # load model
    num_classes = 1
    input_shapes = {k:s_x[k].shape[1:] for k in s_x}
    model = get_particle_net_lite(num_classes, input_shapes, False)

    # if checkpoint directory provided use the latest
    if os.path.isdir(ops.model_weights):
        latest = tf.train.latest_checkpoint(ops.model_weights)
        log.info(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    else:
        model.load_weights(ops.model_weights)

    # make model prediction
    s_p = model.predict(s_x)
    s_p_nodes = tf.reshape(s_p[:,:-1], (-1,8,3))
    s_p_graph = s_p[:,-1]
    # compute mass
    pt = s_x["features"][:,:,0]
    eta = s_x["features"][:,:,1]
    phi = s_x["features"][:,:,2]
    m = s_x["features"][:,:,3]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(m**2 + px**2 + py**2 + pz**2)
    jets = np.stack([e,px,py,pz],axis=-1)
    s_y_parents = np.einsum("bjk, bkn -> bjn", np.transpose(s_y_nodes,[0,2,1]), jets)
    s_y_masses = np.sqrt(s_y_parents[:,:,0]**2 - s_y_parents[:,:,1]**2 - s_y_parents[:,:,2]**2 - s_y_parents[:,:,3]**2)
    s_p_parents = np.einsum("bjk, bkn -> bjn", np.transpose(s_p_nodes,[0,2,1]), jets)
    s_p_masses = np.sqrt(s_p_parents[:,:,0]**2 - s_p_parents[:,:,1]**2 - s_p_parents[:,:,2]**2 - s_p_parents[:,:,3]**2)

    # load background
    b_x, b_weights = get_data.get_data("user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5", True)
    b_p = model.predict(b_x)
    b_p_nodes = tf.reshape(b_p[:,:-1], (-1,8,3))
    b_p_graph = b_p[:,-1]
    # compute mass
    pt = b_x["features"][:,:,0]
    eta = b_x["features"][:,:,1]
    phi = b_x["features"][:,:,2]
    m = b_x["features"][:,:,3]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(m**2 + px**2 + py**2 + pz**2)
    jets = np.stack([e,px,py,pz],axis=-1)
    b_p_parents = np.einsum("bjk, bkn -> bjn", np.transpose(b_p_nodes,[0,2,1]), jets)
    b_p_masses = np.sqrt(b_p_parents[:,:,0]**2 - b_p_parents[:,:,1]**2 - b_p_parents[:,:,2]**2 - b_p_parents[:,:,3]**2)

    # plot
    cut = 0.97
    plt.hist(s_y_masses[:,:2].mean(1),bins=np.linspace(0,5000,100), histtype="step", label="Signal truth", color="black", density=True)
    plt.hist(s_p_masses[:,:2].mean(1),bins=np.linspace(0,5000,100), histtype="step", label="Signal pred", color="blue", density=True)
    plt.hist(b_p_masses[:,:2].mean(1),bins=np.linspace(0,5000,100), histtype="step", label="Bkg pred", color="red", density=True)
    plt.hist(s_p_masses[np.where(s_p_graph > cut)[0]][:,:2].mean(1),bins=np.linspace(0,5000,100), alpha=0.5, label=f"Signal pred w/ {cut} cut", color="blue", density=True)
    plt.hist(b_p_masses[np.where(b_p_graph > cut)[0]][:,:2].mean(1),bins=np.linspace(0,5000,100), alpha=0.5, label=f"Bkg pred w/ {cut} cut", color="red", density=True)
    plt.xlabel("m_{avg} [GeV]")
    plt.ylabel("Density of Events")
    plt.legend()
    plt.savefig("eval.pdf")

    # prepare output file
    # outData = {}
    # outData["p_graph"] = p_graph
    # print(y_nodes[:10])
    # print(p_nodes[:10])
    # print(np.mean(p_graph), np.std(p_graph))

    # # compute parents four mom and masses
    # if not ops.background:
    #     print(y.shape, jets.shape)
    #     y_parents = np.einsum("bjk, bkn -> bjn", np.transpose(y_nodes,[0,2,1]), jets)
    #     y_masses = np.sqrt(y_parents[:,:,0]**2 - y_parents[:,:,1]**2 - y_parents[:,:,2]**2 - y_parents[:,:,3]**2)
    #     plt.hist(y_masses[:,:2].mean(1),bins=np.linspace(0,5000,100), histtype="step", color="black", label="truth", density=True)
    #     plt.hist(y_masses[:,2].flatten(),bins=np.linspace(0,5000,100), alpha=0.5, color="black", label="truth isr", density=True)
    #     outData["mass_true"] = y_masses[:,:2].flatten()
    #     outData["weights_true"] = weights.reshape(-1,1).repeat(2,1).flatten()

    # # compute parents four mom and masses
    # p_parents = np.einsum("bjk, bkn -> bjn", np.transpose(p_nodes,[0,2,1]), jets)
    # p_masses = np.sqrt(p_parents[:,:,0]**2 - p_parents[:,:,1]**2 - p_parents[:,:,2]**2 - p_parents[:,:,3]**2)
    # cut = np.where(p_graph > 0.97)[0]
    # p_masses = p_masses[cut]
    # plt.hist(p_masses[:,:2].mean(1),bins=np.linspace(0,5000,100), histtype="step", color="red", label="prediction", density=True)
    # plt.hist(p_masses[:,2].flatten(),bins=np.linspace(0,5000,100), alpha=0.5, color="red", label="prediction isr", density=True)
    # outData["mass_pred"] = p_masses[:,:2].flatten()
    # outData["weights_pred"] = weights.reshape(-1,1).repeat(2,1).flatten()

    # # save to file
    # for key, val in outData.items():
    #     print(key,val.shape)
    # np.savez("out.npz", **outData)

    # # plot
    # plt.xlabel("m_{avg} [GeV]")
    # plt.ylabel("Density of Events")
    # plt.legend()
    # plt.savefig("eval.pdf")
    # # plt.show()

    

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



