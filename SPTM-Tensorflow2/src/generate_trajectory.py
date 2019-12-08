import habitat
import matplotlib.pyplot as plt
import numpy as np

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def transform_rgb_bgr(image):
	return image[:, :, [2, 1, 0]]


def example():
	trajectory_directory = 'trajectories/Adrian/'
	config = habitat.get_config(config_file='datasets/pointnav/gibson.yaml')
	config.defrost()
	config.DATASET.SPLIT = 'train_mini'
	config.freeze()
	env = habitat.Env(config=config)
	
	action_list = []
	position_list = []
	
	print("Environment creation successful")
	observations = env.reset()
	position_list.append(env.sim.get_agent_state().position)
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(observations["rgb"])
	plt.show(block=False)
	fig.savefig(trajectory_directory+'images/'+str(0)+'.png')
	#cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

	print("Agent stepping around inside environment.")
	
	count_steps = 0
	while not env.episode_over:
		try:
			action=int(input("Enter the action ... "))
		except:
			print('Invalid action')
			continue
		if action > 3:
			print('Invalid action')
			continue
		action_list.append(action)
		observations = env.step(action)
		position_list.append(env.sim.get_agent_state().position)
		count_steps += 1
		ax.imshow(observations["rgb"])
		plt.show(block=False)
		fig.savefig(trajectory_directory+'/images/'+str(count_steps)+'.png')
		#cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
	np.array(action_list).dump(open(trajectory_directory+'actions.npy', 'wb'))
	np.array(position_list).dump(open(trajectory_directory+'positions.npy', 'wb'))
	print("Episode finished after {} steps.".format(count_steps))

if __name__ == "__main__":
	example()


