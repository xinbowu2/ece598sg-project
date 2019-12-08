import pdb 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
from common import *

print("IMPORTS COMPLETE")

  action_mapping = {      
      0: '"Move Forward"',
      1: '"Turn Left"',
      2: '"Turn Right"',
      3: '"Stop"'
  }

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
		if len(batch) == len(images) or len(batch) == 64:
			print(model.test_on_batch(x, y, sample_weight=None, reset_metrics=False))
			batch_x = []
			batch_y = []

def visualize_action_prediction(current_x, future_x, prediction, action):
	fig = plt.figure(figsize=(75,75))
	sub = fig.add_subplot(1,2,1)
	sub.imshow(current_x, interpolation='nearest')
	sub = fig.add_subplot(1,2,2)
	sub.imshow(future_x, interpolation='nearest')
	fig.suptitle('Prediction: ' + action_mapping[prediction] + " Real Action: " + action_mapping[action], fontsize=12)
	fig.savefig('action_prediction.png')

def test_edge_predictor(images, actions, positions):
	model = SiameseResnet(2)
	model.build((32, 256, 256, 6))
	model.load_weights('edge_model.h5')
	adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	batch_x = []
	batch_y = []
	for index1 in range(0, len(images) - 1):
		for index2 in range(index1, len(images) - 1):
			current_x = images[index1]
			future_x = images[index2]

			batch_x.append(np.concatenate((current_x, future_x), axis=2))
			batch_y.append(index2 - index < 8)
			if len(batch) == 64 or (index1 == len(images) - 2 and index2 == len(images) - 2):
				print(model.test_on_batch(x, y, sample_weight=None, reset_metrics=False))
				batch_x = []
				batch_y = []

if __name__ == '__main__':
	print("HELLOOOO")
	trajectory_dir = 'trajectories/Adrian'

	images = []
	image_paths = []
	for im_path in glob.glob(trajectory_dir + "/images/*.png"):
		image_paths.append(im_path)
	image_paths.sort()
	images = [mpimg.imread(x) for x in image_paths]

	actions = np.load(trajectory_dir + '/actions.npy', allow_pickle=True)
	positions = np.load(trajectory_dir + '/positions.npy', allow_pickle=True)

	assert len(images) == len(actions)+1 == len(positions), 'Length of inputs not the same'

	test_action_predictor(images, actions)


