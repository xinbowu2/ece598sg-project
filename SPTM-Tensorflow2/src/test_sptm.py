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

def distance(pos1, pos2):
	return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

action_predictor = ResNet18(3)
action_predictor.build((32, 256, 256, 9))
action_predictor.load_weights('action_model.h5')
action_adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
action_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=action_adam, metrics=['accuracy'])

edge_predictor = SiameseResnet(2)
edge_predictor.build((32, 256, 256, 6))
edge_predictor.load_weights('edge_model.h5')
edge_adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
edge_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=edge_adam, metrics=['accuracy'])

def test_action_predictor(images, actions):
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
			print(action_predictor.evaluate(batch_x, batch_y, batch_size = len(batch_x)))
			batch_x = []
			batch_y = []

def visualize_action_prediction(index1, index2, images, actions):
	if index1 == 0:
		previous_x = images[index1]
	else:
		previous_x = images[index1 - 1]

	current_x = images[index1]
	future_x = images[index2]
	prediction = action_predictor.predict(np.concatenate((previous_x, current_x, future_x), axis=2)).argmax(axis=-1)
	action = actions[index1]
	fig = plt.figure(figsize=(75,75))
	sub = fig.add_subplot(1,2,1)
	sub.imshow(current_x, interpolation='nearest')
	sub = fig.add_subplot(1,2,2)
	sub.imshow(future_x, interpolation='nearest')
	fig.suptitle('Prediction: ' + action_mapping[prediction] + " Real Action: " + action_mapping[action], fontsize=12)
	plt.show()
	fig.savefig('action_prediction.png')

def test_edge_predictor(images, actions, positions):

	batch_x = []
	batch_y = []
	for index1 in range(0, len(images) - 1):
		for index2 in range(index1, len(images) - 1):
			current_x = images[index1]
			future_x = images[index2]

			batch_x.append(np.concatenate((current_x, future_x), axis=2))
			batch_y.append(index2 - index < 8)
			if len(batch) == 64 or (index1 == len(images) - 2 and index2 == len(images) - 2):
				print(edge_predictor.test_on_batch(x, y, sample_weight=None, reset_metrics=False))
				batch_x = []
				batch_y = []

def visualize_edge_predictor(index1, index2, images, positions):
	fig = plt.figure()
	plt.plot(*positions.T, 'b-')
	plt.plot([positions[index1,0], positions[index2,0]], [positions[index1,1], positions[index2,1]],'r-')
	plt.annotate("Start", positions[index1], textcoords="offset points", xytext=(0,10), ha='center')
	plt.annotate("End", positions[index2], textcoords="offset points", xytext=(0,10), ha='center')
	fig.savefig('edge_prediction_trajectory.png')

	fig = plt.figure(figsize=(75,75))
	sub = fig.add_subplot(1,2,1)
	sub.set_title('Start')
	sub.imshow(images[index1], interpolation='nearest')
	sub = fig.add_subplot(1,2,2)
	sub.set_title('End')
	sub.imshow(images[index2], interpolation='nearest')
	# fig.suptitle('Distance: ' + str(distance(positions[index1], positions[index2])), fontsize=12)
	fig.savefig('edge_prediction_images.png')

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
	positions = np.array([positions[:,2], positions[:,0]]).T
	assert len(images) == len(actions)+1 == len(positions), 'Length of inputs not the same'

	# test_action_predictor(images, actions)
	visualize_action_prediction(50, 51, images, actions)


