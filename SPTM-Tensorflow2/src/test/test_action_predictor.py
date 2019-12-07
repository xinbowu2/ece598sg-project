from train.train_setup import *
import pdb
import habitat
import random
import matplotlib.pyplot as plt
print("IMPORTS COMPLETE")
if __name__ == '__main__':
    print("HELLOOOO")
    images = []
    actions = []
    positions = []

    assert len(images) == len(actions) == len(positions), 'Length of inputs not the same'

    model = ResNet18(3)
    model.build((32, 256, 256, 9))
    model.load_weights("../experiments/action/experiment1/models/model.000250.h5")
    adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    batch_x = []
    batch_y = []
    for (index in range(0, len(images) - 1)):
        if index > 0:
            previous_x = images[index - 1]
        else:
            previous_x = images[0]
        current_x = images[index]
        future_x = images[index + 1]

        batch_x.append(np.concatenate((previous_x, current_x, future_x), axis=2))
        batch_y.append(actions[index])
        if len(batch) == 64:
            print(model.test_on_batch(x, y, sample_weight=None, reset_metrics=False))
            batch_x = []
            batch_y = []