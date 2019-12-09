import tensorflow as tf
from models.scene_memory import SceneMemory
from models.att_policy import AttentionPolicyNet
import dataset
import copy
#from config.models import *
import collections
import numpy as np
import random
import copy
import pdb

class RL_Agent(tf.keras.Model):

	def __init__(self, environment, optimizer, loss_function, batch_size, training_embedding=False, epsilon=0.6 ,gamma=0.95, num_actions=3, d_model=128):
		super(RL_Agent, self).__init__()
		
		self.action_size = num_actions
		#print(num_actions)
		self.epsilon = epsilon #epsilon-greedy
			
		self.batch_size = batch_size
		self.training_embedding = training_embedding
		self.scene_memory = SceneMemory()
		self.policy_network = AttentionPolicyNet(num_actions, d_model)
		self.target_policy_network = AttentionPolicyNet(num_actions, d_model)
		self.environment = environment
		self.optimizer = optimizer
		self.loss = loss_function
		self.loss_buffer = None
		if training_embedding:
			self.experience_replay = collections.deque(maxlen=2000*batch_size)
		else:
			self.experience_replay = collections.deque(maxlen=2000)

		self.action_list = []
		self.reward_list = []
		# Initialize discount and exploration rate
		self.gamma = gamma

		self.curr_observations = None
		#self.policy_network.compile(loss='mse', optimizer=self.optimizer)
	
	def trace_model(self):
		observations = self.environment.reset()
		self.update_scene_memory(observations, timestep=[0.0])
		q_vals = self.policy_network(self.scene_memory.obs_embedding, tf.stack(self.scene_memory.memory, axis=1))
		#self.policy_network.build((2,10,128))
		self.policy_network.summary()
		self.scene_memory.summary()
		self.reset()
		

	def update_scene_memory(self, observations, timestep):
		#timestep = [0.0]
		self.scene_memory(observations, timestep=timestep, training_embedding=self.training_embedding)

	#resets the environment and sets current_episode number to r
	def reset(self, episode_index=None):
		self.scene_memory.reset()
		#self.environment.get_env()._current_episode = self.environment.get_env().episodes[episode_idx]

		if episode_index is not None:
			self.environment.get_env()._current_episode_index = episode_index

		self.curr_observations = self.environment.reset()
		self.update_scene_memory(self.curr_observations, timestep=[0.0]) 

		self.action_list = []
		self.reward_list = []

		return copy.deepcopy(self.curr_observations)

	#choose which action to take using epsilon-greedy policy
	def sample_action(self, training=False, evaluating=False):
		if evaluating: 
			q_vals = self.policy_network(self.scene_memory.obs_embedding, tf.stack(self.scene_memory.memory, axis=1))
			#pdb.set_trace()
			action = tf.keras.backend.get_value(tf.keras.backend.argmax(q_vals[:,0,:], 1)[0])
			print(action)
			return action
		elif training and np.random.rand() > self.epsilon: 
			q_vals = self.policy_network(self.scene_memory.obs_embedding, tf.stack(self.scene_memory.memory, axis=1)) #shape is batch_size*1*num_actions
			#print(q_vals)
			action = tf.keras.backend.get_value(tf.random.categorical(q_vals[:,0,:], 1)[0][0])
			#pdb.set_trace()
			if action > self.action_size -1:
				print('invalid action %i'%action)
				return random.randint(0, self.action_size-1)
			else:
				#print(q_vals[:,0,:])
				#if training:
				#print(action)
				return action
		else:
			#print('action size: ', self.action_size)
			return random.randint(0, self.action_size-1) #?? why -2 here

	#returns the observation after taking the action
	def step(self, action, timestep, batch_size, training=False, evaluating=False):
		timestep = float(timestep)
		#current_x = self.environment.step(action)['rgb']/255.0
		self.environment.set_action(action)
		new_observations = self.environment.advance_action()
		self.update_scene_memory(new_observations, timestep=[timestep+1.0])
		
		reward = self.environment.get_reward()
		if not evaluating:
			self.action_list.append(action)
			self.reward_list.append(reward)
			if training:
				if self.training_embedding:
					self.update_model_embedding(batch_size) #train the embeddings from replay buffer 				
				else:
					self.update_model(len(self.scene_memory.memory)-2, batch_size) # discard the last image? 

			if self.training_embedding:
				self.store_observations(timestep, self.curr_observations, action, reward, new_observations, self.environment.is_terminated())		
			#print(self.environment.is_terminated())
			if self.environment.is_terminated():
				if not self.training_embedding:
					self.store_episode(tf.stack(self.scene_memory.memory, axis=1), self.action_list, self.reward_list)

				self.action_list = []
				self.reward_list = []

		self.curr_observations = new_observations
		#print('current reward: ', reward)
		return reward, copy.deepcopy(self.curr_observations)

	def store_observations(self, timestep, curr_observations, action, reward, next_observations, done):
		self.experience_replay.append((timestep, curr_observations, action, reward, next_observations, done))

	#stores a episode in the replay-buffer
	def store_episode(self, memory, action_list, reward_list):
		#print('storing episode')
		self.experience_replay.append((memory, action_list, reward_list))

	def align_target_model(self):
		self.target_policy_network.set_weights(self.policy_network.get_weights())
	
	def update_model_embedding(self, batch_size):
		with tf.GradientTape() as tape:
			batch_sample = random.sample(self.experience_replay, batch_size) #list of tuples - each tuple has (obs, action, reward, next_obs, done)
			timestep = tf.stack([x[0] for x in batch_sample])
			#print('___________________________________________________', timestep)
			observations, next_observations = {}, {}
			for modality in self.scene_memory.modalities:
				observations[modality] = tf.stack([x[1][modality] for x in batch_sample], axis=0)
				next_observations[modality] = tf.stack([x[4][modality] for x in batch_sample], axis=0)
			#observations = tf.stack([x[1] for x in batch_sample])
			#next_observations = tf.stack([x[4] for x in batch_sample])


			embeddings = self.scene_memory.forward_pass(observations, timestep)
			next_embeddings = self.scene_memory.forward_pass(next_observations, timestep)

			action_batch = tf.stack([x[2] for x in batch_sample])
			reward_batch = tf.stack([x[3] for x in batch_sample])
			done = tf.dtypes.cast(tf.stack([x[5] for x in batch_sample]), tf.float32)

			#only consider current embedding in the memory
			q_vals = self.policy_network(embeddings, embeddings)[:,0,:]
			target = copy.deepcopy(q_vals)			
			#print('target shape after copy: ', target.shape)
			#print('target after copy: ', target)
			indices = tf.stack([tf.range(batch_size), action_batch], axis=1)
			#print('indices shape: ', indices.shape)
			#print('indices: ', indices)
			t = self.target_policy_network(next_embeddings, next_embeddings)[:,0,:] #? what is shape of t? batch_size*actions
			#print('t: ', t)
			updates = reward_batch + self.gamma*tf.math.multiply(tf.math.reduce_max(t, axis=1), (1-done))
			#print('updates: ', updates)
			#print('target before scatter: ', target)			
			target = tf.tensor_scatter_nd_update(target, indices, updates)
			#print('target after update: ', target)
			#print('target after scatter: ', target)
			loss = self.loss(target, q_vals)
			#tape.watch(state_batch)
		trainable_variables = self.policy_network.trainable_variables
		trainable_variables.extend(self.scene_memory.trainable_variables) 
		
		# freeze embedding networks when training the policy network
		#print(loss)
		'''
		if self.loss_buffer == None:
			self.loss_buffer = tf.reduce_mean(loss)
		else:
			#print('mean loss difference: ', tf.reduce_mean(loss)-self.loss_buffer)
			self.loss_buffer = tf.reduce_mean(loss)
		'''
		#print('mean loss: ', tf.reduce_mean(loss))
		
		gradients = tape.gradient(loss, trainable_variables)
		#pdb.set_trace()
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))
		#print('mean gradients: ', tf.reduce_mean([tf.reduce_mean(gradient) for gradient in gradients]))
		#pdb.set_trace()
		#print(tf.reduce_sum(loss))
		#pdb.set_trace()
		
	def update_model(self, time_step, batch_size):
		with tf.GradientTape() as tape:
			#print(batch_size)
			#print(len(self.experience_replay))
			batch_sample = random.sample(self.experience_replay, batch_size) 
			minibatch = tf.concat([x[0] for x in batch_sample], axis=0) #batch of memory for full episode

			#size of minibatch should be (batch_size, Horizon, 128)
			horizon = minibatch.shape[1]  #time goes from 0 to horizon-1
		
			memory_batch = minibatch[:,0:time_step+1,:] #hold out memory from 0 to time_step both included

			state_batch = minibatch[:,time_step,:]
			if time_step < horizon-2:
				next_state_batch = minibatch[:,time_step+1,:]

				next_memory_batch = minibatch[:, 0:time_step+2, :]

			#action_batch should be of size batch_size*1
			action_batch = tf.convert_to_tensor([x[1] for x in batch_sample])[:, time_step]
			reward_batch = tf.convert_to_tensor([x[2] for x in batch_sample])[:, time_step]
			q_vals = self.policy_network(state_batch, memory_batch)[:,0,:]
			target = copy.deepcopy(q_vals)


			indices = tf.stack([tf.range(batch_size), action_batch], axis=1)
			if time_step == horizon-2:
				updates = reward_batch
			else:
				#print('current timstep: ', time_step)
				t = self.target_policy_network(next_state_batch, next_memory_batch)[:,0,:] #? what is shape of t? batch_size*actions
				#print(reward_batch)
				updates = reward_batch + self.gamma*tf.math.reduce_max(t, axis=1)
		
			target = tf.tensor_scatter_nd_update(target, indices, updates)
		
			loss = self.loss(target, q_vals)
			#tape.watch(state_batch)
		trainable_variables = self.policy_network.trainable_variables
		#trainable_variables.extend(self.scene_memory.trainable_variables)
		# freeze embedding networks when training the policy network
		gradients = tape.gradient(loss, trainable_variables)
		#pdb.set_trace()
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))

		#pdb.set_trace()
		#print(tf.reduce_sum(loss))
