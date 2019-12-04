from train.train_setup import *

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