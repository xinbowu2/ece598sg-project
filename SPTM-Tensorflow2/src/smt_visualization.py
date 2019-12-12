import pdb 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
from common import *
#import cv2
import habitat

print("IMPORTS COMPLETE")

action_mapping = {      
	0: '"Move Forward"',
	1: '"Turn Left"',
	2: '"Turn Right"',
	3: '"Stop"'
}

def create_video(images): 
    print(images[0].shape)
    size = (images[0].shape[1], images[0].shape[0])    
    
    out = cv2.VideoWriter('trajectory_vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    
    for image in images:
        out.write(image)
    out.release()

def plot_path(positions):
	fig = plt.figure()
	plt.plot(*positions.T, 'b-')
	# plt.plot([positions[index1,0], positions[index2,0]], [positions[index1,1], positions[index2,1]],'r-')
	# plt.annotate("Start", positions[index1], textcoords="offset points", xytext=(0,10), ha='center')
	# plt.annotate("End", positions[index2], textcoords="offset points", xytext=(0,10), ha='center')
	plt.show()
	fig.savefig('trajectory_plot.png')

def avg_dist(positions):
	sum_dist = 0
	num_samples = 0
	for index in range(len(positions) - 5):
		sum_dist += distance(positions[index], positions[index + 5])
		num_samples += 1

	return sum_dist / num_samples

def generate_trajectory(trajectory_directory, actions):
    images = []
    positions = []

    config = habitat.get_config(config_file='datasets/pointnav/gibson.yaml')
    config.defrost()
    config.DATASET.SPLIT = 'train_mini'
    config.freeze()
    env = habitat.Env(config=config)

    print("Environment creation successful")
    observations = env.reset()
    observations = env.reset()
    positions.append(env.sim.get_agent_state().position)
    images.append(observations["rgb"])
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(observations["rgb"])
    plt.show(block=False)
    fig.savefig(trajectory_directory+'images/'+str(0)+'.png')

    print("Agent stepping around inside environment.")
    count_steps = 0
    for action in actions:
        observations = env.step(action)
        positions.append(env.sim.get_agent_state().position)
        images.append(observations["rgb"])
        ax.imshow(observations["rgb"])
        plt.show(block=False)
        fig.savefig(trajectory_directory+'/images/'+str(count_steps)+'.png')
        count_steps += 1

    print("Episode finished after {} steps.".format(count_steps))	

    positions = np.array(positions)
    return images, np.array([positions[:,2], positions[:,0]]).T

if __name__ == '__main__':
    print("HELLOOOO")
    trajectory_directory = 'trajectories/smt/Adrian/'

    actions = [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 2, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1, 0, 2, 1, 1, 2, 2, 1, 0, 0, 2, 2, 1, 1, 1, 2, 0]

    images, positions = generate_trajectory(trajectory_directory, actions)

    plot_path(positions)

    #images = []
    #image_paths = []
    #for im_path in glob.glob(trajectory_directory + "/images/*.png"):
    #    image_paths.append(im_path)
    #image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #images = [cv2.imread(x) for x in image_paths][:-1]

    #create_video(images)
