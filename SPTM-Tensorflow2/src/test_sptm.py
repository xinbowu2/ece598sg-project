import pdb 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
from common import *

print("IMPORTS COMPLETE")

def test_action_predictor(images, actions):
	model = ResNet18(3)
	model.build((32, 256, 256, 9))
	model.load_weights('action_model.h5')
	adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	batch_x = []
	batch_y = []
	for index in range(0, len(images) - 1):
		if index > 0:
			previous_x = images[index - 1]
		else:
			previous_x = images[0]
		current_x = images[index]
		future_x = images[index + 1]
		batch_x.append(np.concatenate((previous_x, current_x, future_x), axis=2))
		batch_y.append(actions[index])
		if index == len(images) - 2:
			print('HELLO')
			batch_x = tf.stack(batch_x, axis=0)
			batch_y = tf.stack(batch_y, axis=0)
			pdb.set_trace()
			print(model.evaluate(batch_x, batch_y, batch_size = len(batch_x)))
			batch_x = []
			batch_y = []

if __name__ == '__main__':
	print("HELLOOOO")
	trajectory_dir = 'trajectories/Adrian'

	images = []
	image_paths = []
	for im_path in glob.glob(trajectory_dir + "/images/*.png"):
		image_paths.append(im_path)
	image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	print(image_paths)
	images = [center_crop_resize(plt.imread(x)[:,:,:3], 256) for x in image_paths][:-1]
	actions = np.load(trajectory_dir + '/actions.npy', allow_pickle=True)[:-1]
	positions = np.load(trajectory_dir + '/positions.npy', allow_pickle=True)[:-1]
	assert len(images) == len(actions)+1 == len(positions), 'Length of inputs not the same'

	test_action_predictor(images, actions)


