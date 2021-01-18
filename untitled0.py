import keras
import tensorflow as tf
from keras.layers.convolutional import Conv2D, AtrousConvolution2D
import torch
import numpy as np

def keras_to_pyt(km, pm=None):
	weight_dict = dict()
	for layer in km.layers:
		if (type(layer) is Conv2D) or (type(layer) is AtrousConvolution2D):
			weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
			weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
		elif type(layer) is keras.layers.Dense:
			weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
			weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
	if pm:
		pyt_state_dict = pm.state_dict()
		for key in pyt_state_dict.keys():
			pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
		pm.load_state_dict(pyt_state_dict)
		return pm
	return weight_dict


#%%
unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)
test=keras_to_pyt(unet)