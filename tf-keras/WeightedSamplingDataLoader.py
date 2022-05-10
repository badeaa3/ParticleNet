import numpy as np
import h5py
import tensorflow as tf

def getXY(inFileName, eventList, signal):
    with h5py.File(inFileName, "r") as hf:

        # if event list is int then choose random
        if isinstance(eventList,int):
            eventList = np.random.random_integers(0,np.array(hf["source"]["mass"]).shape[0]-1,eventList)

        # pick up the kinematics
        m = np.array(hf["source"]["mass"])[eventList]
        pt = np.array(hf["source"]["pt"])[eventList]
        eta = np.array(hf["source"]["eta"])[eventList]
        phi = np.array(hf["source"]["phi"])[eventList]

        # remove spurious nan's in mass
        notNan = np.where(m.sum(1) >= 0)[0]
        m = m[notNan]
        pt = pt[notNan]
        eta = eta[notNan]
        phi = phi[notNan]

        # make graph
        points = np.stack([eta,phi],axis=-1)
        features = np.stack([pt,eta,phi,m], axis=-1)
        mask = np.expand_dims(pt > 0,axis=-1)

        # pick up nodes and graph label
        if signal:
            g = [np.zeros((pt.shape[0],pt.shape[1])),np.zeros((pt.shape[0],pt.shape[1]))]
            for i in [1,2]:
                for j in [1, 2, 3]:
                    a = np.array(hf[f"g{i}"][f"q{j}"])[eventList]
                    g[i-1][np.where(a!=-1)[0],a[np.where(a!=-1)[0]]] = 1
            g = np.stack(g,-1)
            isr = np.expand_dims(np.invert(g.sum(-1).astype(bool)).astype(int),-1)
            nodes = np.concatenate([g,isr],-1)
            nodes = nodes.reshape(nodes.shape[0],-1)
            graph = np.ones((nodes.shape[0],1))
        else:
            nodes = np.zeros((points.shape[0], points.shape[1] * 3))
            graph = np.zeros((points.shape[0], 1))

        y = np.concatenate([nodes,graph],-1)

    return points, features, mask, y

def loadWeightSamples(loadWeightSampler, fileList):
    
    if loadWeightSampler:
        with np.load("WeightSamplerDijets.npz",allow_pickle=True) as x:
            weights = x["weights"]
            fileidx = x["fileidx"]
        usedFiles = fileList
    else:
        weights = []
        fileidx = []
        usedFiles = []
        for iF, file in enumerate(fileList):
            print(f"File {iF}/{len(fileList)}")
            with h5py.File(file, "r") as hf:
                normweight = np.array(hf["normweight"]["normweight"]).flatten()
                nevents = normweight.shape[0]
                if nevents > 0:
                    weights.append(normweight)
                    fileidx.append(np.stack([np.full((nevents),iF),np.arange(nevents)],-1))
                    usedFiles.append(file)
        weights = np.concatenate(weights)
        fileidx = np.concatenate(fileidx)
        np.savez("WeightSamplerDijets.npz",**{"weights":weights,"fileidx":fileidx})
    probabilities = weights / weights.sum()

    return probabilities, fileidx, usedFiles

def formInput(points, features, mask, y):
    x = {
        "points" : points,
        "features" : features,
        "mask" : mask
    }
    return x,y

class WeightedSamplingDataLoader(tf.data.Dataset):

    def _generator(signal, probabilities, fileidx, fileList, num_batches, batch_size):
        for iB in range(num_batches):

            # create batch
            half_batch_size = int(batch_size/2)

            # sample from signal
            points, features, mask, y = getXY(signal, half_batch_size, True)

            # sample from background
            idx = np.random.choice(range(0,len(probabilities)), size=half_batch_size, p = probabilities, replace = True)
            samples = fileidx[idx]
            samples = samples[samples[:, 0].argsort()]
            # get unique files and list of events per file
            files = np.unique(samples[:,0])
            events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
            for iF,iE in zip(files,events):

                b_points, b_features, b_mask, b_y = getXY(fileList[iF], iE, False)
                points = np.concatenate([points, b_points])
                features = np.concatenate([features, b_features])
                mask = np.concatenate([mask, b_mask])
                y = np.concatenate([y, b_y])

            # shuffle
            shuf = np.random.permutation(y.shape[0])
            points = points[shuf]
            features = features[shuf]
            y = y[shuf]

            yield points, features, mask, y

    def __new__(self, njets, signal, probabilities, fileidx, fileList, num_batches, batch_size):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature = (
                tf.TensorSpec(shape = (None, njets, 2), dtype = tf.float64),
                tf.TensorSpec(shape = (None, njets, 4), dtype = tf.float64),
                tf.TensorSpec(shape = (None, njets, 1), dtype = tf.float64),
                tf.TensorSpec(shape = (None, njets*3 + 1), dtype = tf.float64),
            ),
            args=(signal, probabilities, fileidx, fileList, num_batches, batch_size,)
        )

if __name__ == "__main__":
    # FileList = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/abadea/ParticleNet/v1/Dijets/list.txt"
    # fileList = sorted([line.strip() for line in open(FileList,"r")])
    fileList = [
        "/Users/anthonybadea/Documents/ATLAS/rpvmj/ParticleNet/tf-keras/user.jbossios.364708.e7142_e5984_s3126_r10724_r10726_p4355.27261077._000017.trees_expanded_spanet.h5",
        "/Users/anthonybadea/Documents/ATLAS/rpvmj/ParticleNet/tf-keras/user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5"
    ]

    # create data loader
    njets = 8
    signal = "signal_1500_UDB_UDS_training_v65.h5"
    load = True
    probabilities, fileidx, usedFiles = loadWeightSamples(load, "WeightSamplerDijets.npz" if load else fileList)
    num_batches = 5
    batch_size = 4

    dataset = WeightedSamplingDataLoader(njets, signal, probabilities, fileidx, fileList if load else usedFiles, num_batches, batch_size).map(formInput).prefetch(tf.data.AUTOTUNE)

    num_epochs = 1
    for epoch_num in range(num_epochs):
            for x,y in dataset:
                print(y[:,-1])
                print([(key, val.shape) for key,val in x.items()], y.shape)
