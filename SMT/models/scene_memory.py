import tensorflow as tf
from models.resnet18 import ModifiedResNet18
import pdb
import copy

class SceneMemory(tf.keras.Model):
	def __init__(self, modalities=['rgb', 'prev_action'], modality_dim={'rgb':64, 'prev_action':64}, downsampling_size=(64,64), 
	  reduce_factor=4, observation_dim=128):
		super(SceneMemory, self).__init__()
		self.downsampling_size = downsampling_size
		self.modalities = modalities
		self.modality_dim = modality_dim
		self.observation_dim = observation_dim # the dimension of embedding for observations
		self.reduce_factor = reduce_factor
		self.embedding_nets = dict()
		self.memory = []
		self.obs_embedding = None

		if 'rgb' in modalities:
			self.embedding_nets['rgb'] = ModifiedResNet18(modality_dim['rgb'], reduce_factor)

		if 'pose' in modalities:
			self.embedding_nets['pose'] = tf.keras.layers.Dense(modality_dim['pose'])

		if 'prev_action' in modalities:
			self.embedding_nets['prev_action'] = tf.keras.layers.Dense(modality_dim['prev_action'])


		self.fc = tf.keras.layers.Dense(observation_dim)
	
	def reset(self):
		self.obs_embedding = None
		self.memory = []
		return

	#returns the embedding
	def forward_pass(self, observations, timestep, training=False):
		observations = copy.deepcopy(observations) 
		for modality in self.modalities:
			if len(observations[modality].shape) == 3 or len(observations[modality].shape) == 1:
				observations[modality] = tf.expand_dims(observations[modality], 0)
		#print(observations['rgb'].shape)
		#observations['rgb'] = tf.image.resize(observations['rgb'],
		#size=self.downsampling_size)
		
		#observations['rgb'] = tf.transpose(observations['rgb'], perm=[0, 3, 1, 2])
		#if 'prev_action' in self.modalities:
			#observations['prev_action'] = tf.convert_to_tensor(observations['prev_action'], dtype=tf.float32)		
		curr_embedding = self._embed(observations, timestep, training)
		
		return curr_embedding

	def call(self, observations, timestep, training_embedding=False):
		curr_embedding  = self._update(observations, timestep, training_embedding)
		self.obs_embedding = curr_embedding
		return curr_embedding, tf.stack(self.memory, axis=1)

	def _update(self, observations, timestep, training_embedding=False):
		#observations['image'] should be a 4D tensor (batch, height, width, channels)
		
		#input will be (height, width, channels)
		curr_embedding = self.forward_pass(observations, timestep, training=training_embedding)

		if training_embedding: 
			self.memory = [curr_embedding]
		else:
			self.memory.append(curr_embedding)

		return curr_embedding


	def _embed(self, observations, timestep, training=False):
		#print('______________________', timestep)
		'''
		temporal_embedding = tf.math.exp(-tf.convert_to_tensor(timestep,dtype=tf.float32))
		if temporal_embedding.shape[0] == 1:
			temporal_embedding = tf.expand_dims(temporal_embedding, 0)
		else:
			temporal_embedding = tf.reshape(temporal_embedding, [-1,1])
		'''
		embeddings = []
		for modality in self.modalities:
			if modality == 'rgb':
				embeddings.append(self.embedding_nets[modality](observations[modality], training))
			else:
				'''
				concat_embedding = tf.concat([observations[modality], temporal_embedding], axis=1)
				embeddings.append(self.embedding_nets[modality](concat_embedding))
				'''
				embeddings.append(self.embedding_nets[modality](observations[modality]))


		return self.fc(tf.concat(embeddings, axis=1))
