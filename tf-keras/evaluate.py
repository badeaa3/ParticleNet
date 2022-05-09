'''
Authors: Anthony Badea
Date: May 9, 2022
'''

# python imports
import os
import numpy as np
import argparse

# multiprocessing
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# Tensorflow GPU settings
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # force CPU usage

# custom code
from model import get_particle_net, get_particle_net_lite
import get_data

def main():

    # user options
    ops = options()

    # check if multiprocessing should be done
    data_list = handleInput()

    # make output dir
    if ops.outDir:
        outDir = ops.outDir
    else:
        outDir = os.path.join(ops.model_weights.split("/Model")[0], "Dijets" if ops.background else "Signal")
    if not os.path.isdir(outDir):
        os.makedirs(outDir, exist_ok=True)

    # create evaluation job dictionaries
    config  = []
    for data in data_list:
        save_path = os.path.join(outDir, os.path.basename(data).replace(".h5","_eval.npz"))
        config.append({"model_weights" : ops.model_weights,
                       "inFile"    : data,
                       "save_path" : save_path,
                       "background": ops.background})

    # launch jobs
    if len(config) == 1:
        evaluate(config[0])
    else:
        results = mp.Pool(ops.ncpu).map(evaluate, config)

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default=None)
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    parser.add_argument("-b",  "--background", action="store_true")
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    return parser.parse_args()

def handleInput():
    ops = options()
    data = ops.inFile

    # if only one core just return the data
    if ops.ncpu == 1:
        return [data]
    # otherwise return 
    elif os.path.isfile(data) and ".npz" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        from glob import glob
        return sorted(glob(data))
    return []

def evaluate(config):

    ops = options()

    # load data
    if config["background"]:
        x, weights = get_data.get_data(config["inFile"], True)
    else:
        x_train, y_train, weights_train, x_test, y_test, weights_test = get_data.get_data(config["inFile"])
        points = np.concatenate([x_train["points"], x_test["points"]])
        features = np.concatenate([x_train["features"], x_test["features"]])
        mask = np.concatenate([x_train["mask"], x_test["mask"]])
        # combine signal targets
        y_nodes = np.concatenate([y_train, y_test])
        y_graph = np.ones((y_nodes.shape[0],1))
        weights = np.concatenate([weights_train, weights_test])
        x = {
            "points" : points,
            "features" : features,
            "mask" : mask
        }

    # load model
    num_classes = 1
    input_shapes = {k:x[k].shape[1:] for k in x}
    model = get_particle_net_lite(num_classes, input_shapes, False)
    model.load_weights(ops.model_weights)

    # make model prediction
    p = model.predict(x)
    p_nodes = tf.reshape(p[:,:-1], (-1,8,3))
    p_graph = p[:,-1]

    # compute mass
    pt = x["features"][:,:,0]
    eta = x["features"][:,:,1]
    phi = x["features"][:,:,2]
    m = x["features"][:,:,3]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(m**2 + px**2 + py**2 + pz**2)
    jets = np.stack([e,px,py,pz],axis=-1)

    # prepare output file
    outData = {}
    outData["p_graph"] = p_graph

    # compute parents four mom and masses
    if not config["background"]:
        y_parents = np.einsum("bjk, bkn -> bjn", np.transpose(y_nodes,[0,2,1]), jets)
        y_masses = np.sqrt(y_parents[:,:,0]**2 - y_parents[:,:,1]**2 - y_parents[:,:,2]**2 - y_parents[:,:,3]**2)
        outData["mass_true"] = y_masses[:,:2].flatten()
        outData["weights_true"] = weights.reshape(-1,1).repeat(2,1).flatten()

    # compute parents four mom and masses
    p_parents = np.einsum("bjk, bkn -> bjn", np.transpose(p_nodes,[0,2,1]), jets)
    p_masses = np.sqrt(p_parents[:,:,0]**2 - p_parents[:,:,1]**2 - p_parents[:,:,2]**2 - p_parents[:,:,3]**2)
    outData["mass_pred"] = p_masses[:,:2].flatten()
    outData["weights_pred"] = weights.reshape(-1,1).repeat(2,1).flatten()

    # save to file
    np.savez(config["save_path"], **outData)

if __name__ == "__main__":
    main()



