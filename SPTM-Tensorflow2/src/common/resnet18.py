import tensorflow as tf
from common.residual_block import build_res_block_1

NUM_CLASSES=2
class ResNet18(tf.keras.Model):

    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNet18, self).__init__()

        self.pre1 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding='same')
        self.pre2 = tf.keras.layers.BatchNormalization()
        self.pre3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.pre4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2)

        self.layer1 = build_res_block_1(filter_num=64,
                                        blocks=2)
        self.layer2 = build_res_block_1(filter_num=128,
                                        blocks=2,
                                        stride=2)
        self.layer3 = build_res_block_1(filter_num=256,
                                        blocks=2,
                                        stride=2)
        self.layer4 = build_res_block_1(filter_num=512,
                                        blocks=2,
                                        stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        #self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)
        self.fc = tf.keras.layers.Dense(units=num_classes, activation= 'softmax')

    def call(self, inputs, training=None, mask=None):
        pre1 = self.pre1(inputs)
        pre2 = self.pre2(pre1, training=training)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        l1 = self.layer1(pre4, training=training)
        l2 = self.layer2(l1, training=training)
        l3 = self.layer3(l2, training=training)
        l4 = self.layer4(l3, training=training)
        avgpool = self.avgpool(l4)
        out = self.fc(avgpool)

        return out
