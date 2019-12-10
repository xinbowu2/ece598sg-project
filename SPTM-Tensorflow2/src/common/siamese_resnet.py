import tensorflow as tf
from common.resnet18 import ResNet18
import numpy as np
import pdb

NUM_EMBEDDING = 512 #256 #512 #1024 #256 #1024 #256
TOP_HIDDEN = 4 #1 #4
NUM_CLASSES = 2

class SiameseResnet(tf.keras.Model):

	def __init__(self, num_classes=NUM_CLASSES):
		super(SiameseResnet, self).__init__()
		self.res = ResNet18(NUM_EMBEDDING) #512 size embedding of input

		self.bn = tf.keras.layers.BatchNormalization()
		
		self.fc1 = tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal')
		self.bn1 = tf.keras.layers.BatchNormalization()
		
		self.fc2 = tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal')
		self.bn2 = tf.keras.layers.BatchNormalization()
		
		self.fc3 = tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal')
		self.bn3 = tf.keras.layers.BatchNormalization()

		self.fc4 = tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal')
		self.bn4 = tf.keras.layers.BatchNormalization()
		
		self.fc5 = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax, kernel_initializer='he_normal')

	def call(self, inputs, training=None, mask=None):
		first_branch = self.res(inputs[:,:,:,:3]) #embedding for first observation
		second_branch = self.res(inputs[:,:,:,3:]) #embedding for second observation

		raw_result = tf.concat([first_branch, second_branch], axis=1)

		x = self.bn(raw_result)
		x = tf.nn.relu(x)

		x = self.fc1(x)
		x = self.bn1(x)
		x = tf.nn.relu(x)

		x = self.fc2(x)
		x = self.bn2(x)
		x = tf.nn.relu(x)

		x = self.fc3(x)
		x = self.bn3(x)
		x = tf.nn.relu(x)

		x = self.fc4(x)
		x = self.bn4(x)
		x = tf.nn.relu(x)

		x = self.fc5(x)        
		return x


def build_bottom_network(edge_model, input_shape):
	channels, height, width = input_shape
	input = Input(shape=(height, width, channels))
	branch = edge_model.layers[3]
	output = branch(input)
	if NORMALIZATION_ON:
		output = Lambda(lambda x: K.l2_normalize(x, axis=1))(output) 
	return Model(inputs=input, outputs=output)

def build_top_network(edge_model):
	number_of_top_layers = 3 + TOP_HIDDEN * 3
	input = Input(shape=(2 * NUM_EMBEDDING,))
	output = edge_model.layers[-number_of_top_layers](input) #_top_network(input)
	for index in xrange(-number_of_top_layers + 1, 0):
		output = edge_model.layers[index](output)
	return Model(inputs=input, outputs=output)