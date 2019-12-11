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
		
		self.top = tf.keras.Sequential()
		self.top.add(tf.keras.layers.BatchNormalization())
		self.top.add(tf.keras.layers.Activation('relu'))

		self.top.add(tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal'))
		self.top.add(tf.keras.layers.BatchNormalization())
		self.top.add(tf.keras.layers.Activation('relu'))

		self.top.add(tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal'))
		self.top.add(tf.keras.layers.BatchNormalization())
		self.top.add(tf.keras.layers.Activation('relu'))

		self.top.add(tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal'))
		self.top.add(tf.keras.layers.BatchNormalization())
		self.top.add(tf.keras.layers.Activation('relu'))

		self.top.add(tf.keras.layers.Dense(units=NUM_EMBEDDING,
										 kernel_initializer='he_normal'))
		self.top.add(tf.keras.layers.BatchNormalization())
		self.top.add(tf.keras.layers.Activation('relu'))

		self.top.add(tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax, kernel_initializer='he_normal'))

	def call(self, inputs, training=None, mask=None):
		first_branch = self.res(inputs[:,:,:,:3]) #embedding for first observation
		second_branch = self.res(inputs[:,:,:,3:]) #embedding for second observation

		raw_result = tf.concat([first_branch, second_branch], axis=1)

		x = self.top(raw_result)     
		return x


	def build_bottom_network(self, input_shape):
		channels, height, width = input_shape
		input = tf.keras.layers.Input(shape=(height, width, channels))
		branch = self.res
		output = branch(input)
		if NORMALIZATION_ON:
			output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(output) 
		return tf.keras.Model(inputs=input, outputs=output)

	def build_top_network(self):
		input = tf.keras.layers.Input(shape=(2 * NUM_EMBEDDING,))
		output = self.top(input)
		return tf.keras.Model(inputs=input, outputs=output)
