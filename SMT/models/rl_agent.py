import tensorflow as tf
from models.scene_memory import SceneMemory
from models.att_policy import AttentionPolicyNet
#from config.models import *
import collections
import numpy as np
import random
import copy
import pdb

action_mapping = {      
	  0: 'move_forward',
	  1: 'turn left',
	  2: 'turn right',
	  3: 'stop'
}

batch_size = 5

class RL_Agent(tf.keras.Model):

	def __init__(self, environment, optimizer, loss_function, training_embedding=False, epsilon=0.1 ,gamma=0.95, num_actions=4, d_model=128):
		super(RL_Agent, self).__init__()
		
		self.action_size = num_actions
		self.epsilon = epsilon #epsilon-greedy
			
		self.training_embedding = training_embedding

		self.scene_memory = SceneMemory()
		self.policy_network = AttentionPolicyNet(num_actions, d_model)
		self.target_policy_network = AttentionPolicyNet(num_actions, d_model)
		self.environment = environment
		self.optimizer = optimizer
		self.loss_function = loss_function

		self.experience_replay = collections.deque(maxlen=2000)
		self.action_list = []
		self.reward_list = []
		# Initialize discount and exploration rate
		self.gamma = gamma
		#self.policy_network.compile(loss='mse', optimizer=self.optimizer)

	def update_scene_memory(self, image):
		obs = {}
		obs['image'] = image
		self.scene_memory(obs, training_embedding=self.training_embedding)
		return

	#resets the environment and sets current_episode number to r
	def reset(self, episode_idx):
		self.scene_memory.reset()
		self.environment._current_episode = self.environment.episodes[episode_idx]
		current_x = self.environment.reset()['rgb']/255.0
		self.update_scene_memory(current_x) 
		self.action_list = []
		self.reward_list = []

	#choose which action to take using epsilon-greedy policy
	def sample_action(self):
		if np.random.rand() <= self.epsilon:
			return random.randint(0, len(action_mapping)-2)
		else:
			q_vals = self.policy_network(self.scene_memory.obs_embedding, tf.stack(self.scene_memory.memory, axis=1)) #shape is batch_size*1*num_actions
			return tf.keras.backend.get_value(tf.random.categorical(q_vals[:,0,:], 1)[0][0])

	#returns the observation after taking the action
	def step(self, action, training=False):
		current_x = self.environment.step(action)['rgb']/255.0
		self.update_scene_memory(current_x)
		
		reward = 1.0
		self.action_list.append(action)
		self.reward_list.append(reward)
		if training:
			self.update_model(len(self.scene_memory.memory)-2, batch_size) # discard the last observation? 
		
		if self.environment.episode_over:
			self.store_episode(tf.stack(self.scene_memory.memory, axis=1), self.action_list, self.reward_list)
			self.memory = None
			self.action_list = []
			self.reward_list = []

	#stores a episode in the replay-buffer
	def store_episode(self, memory, action_list, reward_list):
		self.experience_replay.append((memory, action_list, reward_list))

	def align_target_model(self):
		self.target_policy_network.set_weights(self.policy_network.get_weights())
	
	def update_model(self, time_step, batch_size=64):
		with tf.GradientTape() as tape:
			batch_sample = random.sample(self.experience_replay, batch_size) 
			minibatch = tf.concat([x[0] for x in batch_sample], axis=0) #batch of memory for full episode

			#size of minibatch should be (batch_size, Horizon, 128)
			horizon = minibatch.shape[1]-1 #time goes from 0 to horizon-1
			#time_step = np.random.random_integers(1,horizon-1)
		
			memory_batch = minibatch[:,0:time_step+1,:] #hold out memory from 0 to time_step both included

			state_batch = minibatch[:,time_step,:]
			if time_step < horizon-1:
				next_state_batch = minibatch[:,time_step+1,:]

				next_memory_batch = minibatch[:, 0:time_step+2, :]

			#action_batch should be of size batch_size*1
			action_batch = tf.convert_to_tensor([x[1] for x in batch_sample])[:, time_step]
			reward_batch = tf.convert_to_tensor([x[2] for x in batch_sample])[:, time_step]

			q_vals = self.policy_network(state_batch, memory_batch)[:,0,:]
			target = copy.deepcopy(q_vals)


			indices = tf.stack([tf.range(batch_size), action_batch], axis=1)
			if time_step == horizon-1:
				updates = reward_batch
			else:
				t = self.target_policy_network(next_state_batch, next_memory_batch)[:,0,:] #? what is shape of t? batch_size*actions
				updates = reward_batch + self.gamma*tf.math.reduce_max(t, axis=1)
		
			target = tf.tensor_scatter_nd_update(target, indices, updates)
		
			loss = self.loss_function(target, q_vals)
			#tape.watch(state_batch)
		trainable_variables = self.policy_network.trainable_variables
		if self.training_embedding:
			trainable_variables.append(self.scene_memory.trainable_variables) 

		# freeze embedding networks when training the policy network
		gradients = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))

		#pdb.set_trace()
		print(tf.reduce_sum(loss))
