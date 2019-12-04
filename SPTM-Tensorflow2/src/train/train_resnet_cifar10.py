import tensorflow as tf

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, is_first_block_of_first_layer=False):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu = tf.nn.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2, training=training)

        output = tf.nn.relu(tf.keras.layers.add([identity, bn2]))

        return output

def build_res_block_1(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block

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
        self.fc = tf.keras.layers.Dense(units=num_classes)

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

def main():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Instantiate the Dataset class.
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(50000).batch(128, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    model = ResNet18(3)
    adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(train_dataset,
          epochs=20,
          validation_data=test_dataset,
          validation_freq=1)

    
if __name__ == "__main__":
    main()