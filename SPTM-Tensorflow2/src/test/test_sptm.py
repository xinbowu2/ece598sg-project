from train.train_setup import *
import pdb
import habitat
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

print("IMPORTS COMPLETE")

def test_action_predictor(images, actions):
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

def main():

if __name__ == '__main__':
    print("HELLOOOO")
    trajectory_dir = 'trajectories/Adrian'

    images = []
    for im_path in glob.glob(trajectory_dir + "/images/*.png"):
        images.append(mpimg.imread(im_path))
    actions = np.load(trajectory_dir + 'actions.npy') 
    positions = np.load(trajectory_dir + 'positions.npy') 

    assert len(images) == len(actions) == len(positions), 'Length of inputs not the same'

    test_action_predictor(images, actions)

