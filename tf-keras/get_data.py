import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(config):

	with h5py.File(config["inFileName"], "r") as hf:

		# pick up event weights
		weights = np.array(hf["normweight"]["normweight"]).flatten()

		# pick up the kinematics
		m = np.array(hf["source"]["mass"])
		pt = np.array(hf["source"]["pt"])
		eta = np.array(hf["source"]["eta"])
		phi = np.array(hf["source"]["phi"])
		
		# normalize
		# m = (m - m.mean(1).reshape(-1,1))/(m.std(1).reshape(-1,1))
		# pt = (pt - pt.mean(1).reshape(-1,1))/(pt.std(1).reshape(-1,1))

		# make graph
		points = np.stack([eta,phi],axis=-1)
		features = np.stack([pt,eta,phi,m], axis=-1)
		mask = np.expand_dims(pt > 0,axis=-1) # mask = 0 for padded

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

		points_train, points_test, features_train, features_test, mask_train, mask_test, y_train, y_test = train_test_split(points,features,mask,y,test_size=0.33, random_state=42)

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

		return x_train, y_train, x_test, y_test

if __name__ == "__main__":
	x_train, y_train, x_test, y_test = get_data({"inFileName" : "signal_1500_UDB_UDS_training_v65.h5"})
	print([i.shape for key,i in x_train.items()], y_train.shape)
	print([i.shape for key,i in x_test.items()], y_test.shape)