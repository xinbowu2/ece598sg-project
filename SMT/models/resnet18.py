import tensorflow as tf
from residual_block import build_res_block_1

class ModifiedResNet18(tf.keras.Model):

    def __init__(self, embedding_dim=64, reduce_factor=4):
        super(ModifiedResNet18, self).__init__()
        self.embedding_dim = embedding_dim
        self.reduce_factor = reduce_factor

        self.pre1 = tf.keras.layers.Conv2D(filters=64//reduce_factor,
                                           kernel_size=(7, 7),
                                           strides=1,
                                           padding='same')
        self.pre2 = tf.keras.layers.BatchNormalization()
        self.pre3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        # keep maxpool ?? 
        self.pre4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2)

        self.layer1 = build_res_block_1(filter_num=64//reduce_factor,
                                        blocks=2)
        self.layer2 = build_res_block_1(filter_num=128//reduce_factor,
                                        blocks=2,
                                        stride=2)
        self.layer3 = build_res_block_1(filter_num=256//reduce_factor,
                                        blocks=2,
                                        stride=2)
        self.layer4 = build_res_block_1(filter_num=512//reduce_factor,
                                        blocks=2,
                                        stride=2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs, training=None, mask=None):
        pre1 = self.pre1(inputs)
        pre2 = self.pre2(pre1, training=training)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        l1 = self.layer1(pre4, training=training)
        l2 = self.layer2(l1, training=training)
        l3 = self.layer3(l2, training=training)
        l4 = self.layer4(l3, training=training)

        out = self.flatten(l4)
        out = self.fc(out)

        return out