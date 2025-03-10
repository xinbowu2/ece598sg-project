from train.train_setup import *


def preprocess(x, y):
  x = tf.image.per_image_standardization(x)
  x = tf.cast(x, tf.float32)
  return x, y


def main():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Instantiate the Dataset class.
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess).shuffle(50000).batch(128, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(128, drop_remainder=True)

    #base_model = tf.keras.applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (32,32,3))
    #x = base_model.output
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
    #predictions = tf.keras.layers.Dense(10, activation= 'softmax')(x)
    #model = tf.keras.models.Model(inputs = base_model.input, outputs = predictions)
    model = ResNet18(10)
    adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(train_dataset,
          epochs=20,
          validation_data=test_dataset,
          validation_freq=1)

    
if __name__ == "__main__":
    main()
