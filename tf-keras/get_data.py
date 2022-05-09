import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(inFileName, background=False):

	with h5py.File(inFileName, "r") as hf:

		# pick up the kinematics
		m = np.array(hf["source"]["mass"])
		pt = np.array(hf["source"]["pt"])
		eta = np.array(hf["source"]["eta"])
		phi = np.array(hf["source"]["phi"])
		weights = np.array(hf["normweight"]["normweight"]).flatten()
		
		# remove spurious nan's in mass
		notNan = np.where(m.sum(1) >= 0)[0]
		m = m[notNan]
		pt = pt[notNan]
		eta = eta[notNan]
		phi = phi[notNan]
		weights = weights[notNan]

		# normalize
		# m = (m - m.mean(1).reshape(-1,1))/(m.std(1).reshape(-1,1))
		# pt = (pt - pt.mean(1).reshape(-1,1))/(pt.std(1).reshape(-1,1))

		# make graph
		points = np.stack([eta,phi],axis=-1)
		features = np.stack([pt,eta,phi,m], axis=-1)
		mask = np.expand_dims(pt > 0,axis=-1) # mask = 0 for padded

		if background:
			x = {
				"points" : points,
				"features" : features,
				"mask" : mask
			}
			return x, weights

		# pick up indices
		g = [np.zeros((pt.shape[0],pt.shape[1])),np.zeros((pt.shape[0],pt.shape[1]))]
		for i in [1,2]:
			for j in [1, 2, 3]:
				a = np.array(hf[f"g{i}"][f"q{j}"])
				g[i-1][np.where(a!=-1)[0],a[np.where(a!=-1)[0]]] = 1
		g = np.stack(g,-1)
		isr = np.expand_dims(np.invert(g.sum(-1).astype(bool)).astype(int),-1)
		y = np.concatenate([g,isr],-1)

		# print(points.shape, features.shape, mask.shape, y.shape)

		points_train, points_test, features_train, features_test, mask_train, mask_test, y_train, y_test, weights_train, weights_test = train_test_split(points,features,mask,y,weights,test_size=0.33, random_state=42)

		x_train = {
			"points" : points_train,
			"features" : features_train,
			"mask" : mask_train
		}
		x_test = {
			"points" : points_test,
			"features" : features_test,
			"mask" : mask_test
		} 

		return x_train, y_train, weights_train, x_test, y_test, weights_test

def get_signal_and_background(signal,background):

		# load
		s_x_train, s_y_train, s_weights_train, s_x_test, s_y_test, s_weights_test = get_data(signal)
		b_x, b_weights = get_data(background, True)

		# combine signal inputs
		points = np.concatenate([s_x_train["points"], s_x_test["points"]])
		features = np.concatenate([s_x_train["features"], s_x_test["features"]])
		mask = np.concatenate([s_x_train["mask"], s_x_test["mask"]])
		# combine signal targets
		nodes = np.concatenate([s_y_train, s_y_test])
		graph = np.ones((nodes.shape[0],1))
		weights = np.concatenate([s_weights_train, s_weights_test])
		# combine with background
		points = np.concatenate([points, b_x["points"]])
		features = np.concatenate([features, b_x["features"]])
		mask = np.concatenate([mask, b_x["mask"]])
		b_nodes = np.zeros((b_x["points"].shape[0], b_x["points"].shape[1], 3))
		# b_nodes[:,:,2] = 1
		nodes = np.concatenate([nodes, b_nodes])
		graph = np.concatenate([graph, np.zeros((b_x["points"].shape[0], 1))])
		weights = np.concatenate([weights, b_weights])

		# split
		points_train, points_test, features_train, features_test, mask_train, mask_test, nodes_train, nodes_test, graph_train, graph_test, weights_train, weights_test = train_test_split(points,features,mask,nodes,graph,weights,test_size=0.33, random_state=42)

		x_train = {
			"points" : points_train,
			"features" : features_train,
			"mask" : mask_train
		}
		x_test = {
			"points" : points_test,
			"features" : features_test,
			"mask" : mask_test
		}

		y_train = np.concatenate([nodes_train.reshape(nodes_train.shape[0],-1),graph_train],-1)
		y_test = np.concatenate([nodes_test.reshape(nodes_test.shape[0],-1),graph_test],-1)

		return x_train, y_train, weights_train, x_test, y_test, weights_test


if __name__ == "__main__":
	
	x_train, y_train, weights_train, x_test, y_test, weights_test = get_signal_and_background(signal="signal_1500_UDB_UDS_training_v65.h5", background="user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5")
	print([(key,i.shape) for key,i in x_train.items()], y_train.shape)
	print([(key,i.shape) for key,i in x_test.items()], y_test.shape)

	# x_train, y_train, x_test, y_test = get_data(inFileName = "signal_1500_UDB_UDS_training_v65.h5")
	# print([i.shape for key,i in x_train.items()], y_train.shape)
	# print([i.shape for key,i in x_test.items()], y_test.shape)

	# x = {key : np.concatenate([x_train[key], x_test[key]]) for key in x_train.keys()}
	# y = np.concatenate([y_train, y_test])

    # # pick up the kinematics
	# pt = x["features"][:,:,0]
	# eta = x["features"][:,:,1]
	# phi = x["features"][:,:,2]
	# m = x["features"][:,:,3]

	# # convert to cartesian
	# px = pt * np.cos(phi)
	# py = pt * np.sin(phi)
	# pz = pt * np.sinh(eta)
	# e  = np.sqrt(m**2 + px**2 + py**2 + pz**2)
	# jets = np.stack([e,px,py,pz],axis=-1)

	# # compute parents four mom
	# parents = np.einsum("bjk, bkn -> bjn", np.transpose(y,[0,2,1]), jets)
	# # compute parent masses
	# masses = np.sqrt(parents[:,:,0]**2 - parents[:,:,1]**2 - parents[:,:,2]**2 - parents[:,:,3]**2)
	
	# import matplotlib.pyplot as plt 
	# plt.hist(masses[:,:2].flatten(),bins=np.linspace(0,2500,50), histtype="step", color="blue", label="truth")
	# plt.legend()
	# plt.show()
