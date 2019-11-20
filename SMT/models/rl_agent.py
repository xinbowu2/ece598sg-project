import tensorflow as tf
from models.scene_memory import SceneMemory
from models.att_policy import AttentionPolicyNet
from config.models import *
import numpy as np
import random

action_mapping = {      
      0: 'move_forward',
      1: 'turn left',
      2: 'turn right',
      3: 'stop'
}

batch_size = 64

class RL_Agent(tf.keras.Model):

	def __init__(self, environment, optimizer, loss_function, epsilon=0.1 ,gamma=0.95, num_actions=4, d_model=128):
		super(RL_Agent, self).__init__()
		
		self.action_size = num_actions
		self.epsilon = epsilon #epsilon-greedy
		
		self.scene_memory = SceneMemory()
		self.policy_network = AttentionPolicyNet(num_actions, d_model)
		self.target_policy_network = self.policy_network.clone()
		self.environment = environment
		self.optimizer = optimizer
		self.loss_function = loss_function

		self.obs_embedding = None
		self.memory = None

		self.experience_replay = deque(maxlen=2000)
		self.action_list = []
		self.reward_list = []
		# Initialize discount and exploration rate
		self.gamma = gamma

		#self.policy_network.compile(loss='mse', optimizer=self.optimizer)

	#choose which action to take using epsilon-greedy policy
	def sample_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randint(0, len(action_mapping)-2)
		else:
			q_vals = self.policy_network(self.obs_embedding, self.memory)
			return tf.random.categorical(q_vals, 1)

	#returns the observation after taking the action
	def step(self, action):
		obs = {}
		obs['image'] = self.environment.step(action)['rgb']/255.0
		self.action_list.append(action)
		self.reward_list.append(reward)

		new_obs_embedding, self.memory = self.scene_memory(obs)
		#write a function to calculate the reward 
		self.obs_embedding = new_obs_embedding
		if self.environment.episode_over:
			store_episode(self.memory, self.action_list, self.reward_list)
			self.memory = None
			self.action_list = []
			self.reward_list = []
	#stores a episode in the replay-buffer
	def store_episode(self, memory, action_list, reward_list):
		self.experience_replay.append(memory, action_list, reward_list)

	def align_target_model(self):
    	self.target_policy_network.set_weights(self.policy_network.get_weights())
	
	def update_model(self, batch_size):
		minibatch = random.sample(self.experience_replay, batch_size) #size of minibatch = (batch_size, Horizon, 128)
		action_list = minibatch[:,1]
		reward_list = minibatch[:,2]
		minibatch = minibatch[:,0]
		horizon = minibatch.shape[1]
		time_step = np.random.random_integers(1,horizon-1)
		
		memory_batch = minibatch[:,0:time_step,:]
		state_batch = minibatch[:,time_step-1,:]
		next_state_batch = minibatch[:,time_step,:]
		next_memory_batch = minibatch[:, 0:time_step+1, :]
		action_batch = action_list[time_step-1]
		reward_batch = reward_list[time_step-1]

		target = self.policy_network(state_batch, memory_batch)
		q_vals = target.clone()
		if time_step == horizon-1:
			target[action_batch] = reward_batch
		else:
			t = self.target_policy_network(next_state_batch, next_memory_batch) #? what is shape of t? batch_size*actions
			target[action_batch] = reward_batch + self.gamma*tf.math.reduce_max(t, axis=1)
		
		with tf.GradientTape() as tape:
			loss = self.loss_function(target, q_vals)
		gradients = tape.gradient(loss, self.policy_network.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
