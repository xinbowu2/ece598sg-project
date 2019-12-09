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

MAX_EDGE_DIST = 1
NEG_SAMPLE_MULT = 5

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
	x = []
	y = []
	for index in range(0, len(images) - 1):
		if index > 0:
			previous_x = images[index - 1]
		else:
			previous_x = images[0]
		current_x = images[index]
		future_x = images[index + 1]
		x.append(np.concatenate((previous_x, current_x, future_x), axis=2))
		y.append(actions[index])

	print(action_predictor.evaluate(np.array(x), np.array(y), batch_size=64))

def visualize_action_prediction(index1, index2, images, actions):
	if index1 == 0:
		previous_x = images[index1]
	else:
		previous_x = images[index1 - 1]

	current_x = images[index1]
	future_x = images[index2]
	sample = np.concatenate((previous_x, current_x, future_x), axis=2)
	output = action_predictor.predict(np.array([sample]))[0] 
	prediction = output.argmax()
	
	action = actions[index1]
	fig, ax = plt.subplots(1,2)
	ax[0].set_title('Current')
	ax[0].imshow(current_x, interpolation='nearest')
	ax[1].set_title('Future')
	ax[1].imshow(future_x, interpolation='nearest')
	[axi.set_axis_off() for axi in ax.ravel()]
	fig.suptitle('Prediction: %s (%f)\nReal Action: %s' % (action_mapping[prediction], output[prediction], action_mapping[action]), fontsize=12)
	plt.show()
	fig.savefig('action_prediction.png')

def test_edge_predictor(images, actions, positions):
	x = []
	y = []
	for index1 in range(0, len(images) - 1):
		for index2 in range(index1, len(images) - 1):
			current_x = images[index1]
			future_x = images[index2]

			dist = distance(positions[index1], positions[index2])
			if dist < MAX_EDGE_DIST:
				x.append(np.concatenate((current_x, future_x), axis=2))
				y.append(1)
			elif dist > MAX_EDGE_DIST * NEG_SAMPLE_MULT:
				x.append(np.concatenate((current_x, future_x), axis=2))
				y.append(0)
	 
	print(edge_predictor.evaluate(np.array(x), np.array(y), batch_size=64))

def visualize_edge_prediction(index1, index2, images, positions):
	fig = plt.figure()
	plt.plot(*positions.T, 'b-')
	plt.plot([positions[index1,0], positions[index2,0]], [positions[index1,1], positions[index2,1]],'r-')
	plt.annotate("Start", positions[index1], textcoords="offset points", xytext=(0,10), ha='center')
	plt.annotate("End", positions[index2], textcoords="offset points", xytext=(0,10), ha='center')
	plt.show()
	fig.savefig('edge_prediction_trajectory.png')

	current_x = images[index1]
	future_x = images[index2]
	sample = np.concatenate((current_x, future_x), axis=2)
	output = edge_predictor.predict(np.array([sample]))[0]
	prediction = output.argmax()
	
	if prediction == 1:
		prediction_str = 'Close'
	else:
		prediction_str = 'Far'
	
	fig, ax = plt.subplots(1,2)
	fig.suptitle('Prediction: %s (%f)' % (prediction_str, output[prediction]))
	ax[0].set_title('Start')
	ax[0].imshow(images[index1], interpolation='nearest')
	ax[1].set_title('End')
	ax[1].imshow(images[index2], interpolation='nearest')
	[axi.set_axis_off() for axi in ax.ravel()]
	# fig.suptitle('Distance: ' + str(distance(positions[index1], positions[index2])), fontsize=12)
	plt.show()
	fig.savefig('edge_prediction_images.png')

def avg_dist(positions):
	sum_dist = 0
	num_samples = 0
	for index in range(len(positions) - 5):
		sum_dist += distance(positions[index], positions[index + 5])
		num_samples += 1

	return sum_dist / num_samples
		

if __name__ == '__main__':
	print("HELLOOOO")
	trajectory_dir = 'trajectories/Adrian'

	images = []
	image_paths = []
	for im_path in glob.glob(trajectory_dir + "/images/*.png"):
		image_paths.append(im_path)
	image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	images = [center_crop_resize(plt.imread(x)[:,:,:3], 256) for x in image_paths][:-1]
	actions = np.load(trajectory_dir + '/actions.npy', allow_pickle=True)[:-1]
	positions = np.load(trajectory_dir + '/positions.npy', allow_pickle=True)[:-1]
	positions = np.array([positions[:,2], positions[:,0]]).T
	assert len(images) == len(actions)+1 == len(positions), 'Length of inputs not the same'

	# print(avg_dist(positions))

	#test_action_predictor(images, actions)
	#visualize_action_prediction(0, 1, images, actions)
	# test_edge_predictor(images, actions, positions)
	visualize_edge_prediction(100, 105, images, positions)
