import tensorflow as tf
from config import NUM_CLASSES
from common.resnet18 import ResNet18
import numpy as np

NUM_EMBEDDING = 512 #256 #512 #1024 #256 #1024 #256
TOP_HIDDEN = 4 #1 #4

class SiameseResnet(tf.keras.Model):

    def __init__(self, num_classes=NUM_CLASSES):
        super(SiameseResnet, self).__init__()
        self.res = ResNet18(NUM_EMBEDDING) #512 size embedding of input

        self.bn = tf.keras.layers.BatchNormalization()
        
        self.fc1 = tf.keras.layers.Dense(units=NUM_EMBEDDING)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.fc2 = tf.keras.layers.Dense(units=NUM_EMBEDDING)
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.fc3 = tf.keras.layers.Dense(units=NUM_EMBEDDING)
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.fc4 = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        first_branch = self.res(inputs[:,:,:,:3])
        second_branch = self.res(inputs[:,:,:,3:])

        raw_result = np.concatenate([first_branch, second_branch])

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

        return x