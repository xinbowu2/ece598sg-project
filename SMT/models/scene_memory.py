import tensorflow as tf
from models.resnet18 import ModifiedResNet18
import pdb

class SceneMemory(tf.keras.Model):
	def __init__(self, modalities=['image'], modality_dim={'image':64}, downsampling_size=(64,64), 
	  reduce_factor=4, observation_dim=128):
		super(SceneMemory, self).__init__()
		self.downsampling_size = downsampling_size
		self.modalities = modalities
		self.modality_dim = modality_dim
		self.observation_dim = observation_dim # the dimension of embedding for observations
		self.reduce_factor = reduce_factor
		self.embedding_nets = dict()
		self.memory = []

		if 'image' in modalities:
			self.embedding_nets['image'] = ModifiedResNet18(modality_dim['image'], reduce_factor)

		if 'pose' in modalities:
			self.embeddings_nets['pose'] = tf.keras.layers.Dense(modality_dim['pose'])

		if 'prev_action' in modalities:
			self.embeddings_nets['prev_action'] = tf.keras.layers.Dense(modality_dim['prev_action'])


		self.fc = tf.keras.layers.Dense(observation_dim)
	

	def call(self, observations, training=None, training_embedding=False):
		curr_embedding  = self._update(observations, training, training_embedding)

		return curr_embedding, tf.stack(self.memory, axis=1)


	def _update(self, observations, training=None, training_embedding=False):
		#observations['image'] should be a 4D tensor (batch, height, width, channels)
		#input will be (height, width, channels)
		for modality in self.modalities:
			if len(observations[modality].shape) == 3:
				observations[modality] = tf.expand_dims(observations[modality], 0)

		observations['image'] = tf.image.resize(observations['image'],
		size=self.downsampling_size)

		observations['image'] = tf.transpose(observations['image'], perm=[0, 3, 1, 2])

		curr_embedding = self._embed(observations, training)

    if training_embedding: 
      self.memory = [curr_embedding]
    else:
		  self.memory.append(curr_embedding)

		return curr_embedding


	def _embed(self, observations, training=None):
		embeddings = []

		for modality in self.modalities:
			if modality == 'image':
				embeddings.append(self.embedding_nets[modality](observations[modality], training))
			else:
				embeddings.append(self.embedding_nets[modality](observations[modality]))

		concat_embedding = tf.concat(embeddings, axis=1)

		return self.fc(concat_embedding)





