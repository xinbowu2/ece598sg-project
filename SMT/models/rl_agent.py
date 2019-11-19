import tensorflow as tf
from models.scene_memory import SceneMemory
from models.att_policy import AttentionPolicyNet
from config.models import *
import numpy as np

action_mapping = {      
      0: 'move_forward',
      1: 'turn left',
      2: 'turn right',
      3: 'stop'
}

batch_size = 64

class RL_Agent(tf.keras.Model):

	def __init__(self, enviroment, optimizer, loss_function, epsilon=0.1 ,gamma=0.95, num_actions=4, d_model=1024):
		super(RL_Agent, self).__init__()
		
		self.action_size = num_actions
		self.epsilon = epsilon #epsilon-greedy
		
		self.scene_memory = SceneMemory()
		self.policy_network = AttentionPolicyNet()

		self.enviroment = enviroment
		self.optimizer = optimizer
		self.loss_function = loss_function

		self.obs_embedding = None
		self.memory = None

		self.experience_replay = deque(maxlen=2000)

		# Initialize discount and exploration rate
		self.gamma = gamma

	def store(self, state, action, reward, next_state, memory, terminated):
        self.experience_replay.append((state, action, reward, next_state, memory, terminated))

	#choose which action to take using epsilon-greedy policy
	def sample_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randint(0, len(action_mapping)-2)
		else:
			q_vals = self.policy_network(self.obs_embedding, self.memory)
			return tf.random.categorical(q_vals, 1)[0]

	#returns the observation after taking the action
	def step(self, action):

		obs = {}
		obs['image'] = self.enviroment.step(action)['rgb']/255.0
		new_obs_embedding, self.memory = self.scene_memory(obs)
		#write a function to calculate the reward 

		self.store(self.obs_embedding, action, reward, new_obs_embedding, self.memory, enviroment.episode_over)
		self.obs_embedding = new_obs_embedding


	def retrain(self, batch_size):
	    minibatch = random.sample(self.experience_replay, batch_size)
		states = minibatch[:,0]
		next_states = minibatch[:,3]
		memory = minibatch[:, 4]
		actions = minibatch[:, 1]
		rewards = minibatch[:, 2]
		terminated = minibatch[:, 5]
		q_vals = self.policy_network(states, memory)
		target = q_vals.copy()
		t = self.policy_network(next_states, memory)

		target[:, actions] = rewards + self.gamma* np.max(target, axis=1) *(1-terminated) #verify this??? 

		with tf.GradientTape() as tape:
		    loss = self.loss_function(target, q_vals)
		  	gradients = tape.gradient(loss, #)    
		  	self.optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

	    # for state, action, reward, next_state, terminated in minibatch:

	    #     target = self.q_network.predict(state)

	    #     loss = self.criterion(qvals[action], target)
	    #     loss.backward()
	    #     self.optimizer.step()

	    #     return loss.item()

	    #     if terminated:
	    #         target[0][action] = reward
	    #     else:
	    #         t = self.target_network.predict(next_state)
	    #         target[0][action] = reward + self.gamma * np.amax(t)

	    #     self.q_network.fit(state, target, epochs=1, verbose=0)