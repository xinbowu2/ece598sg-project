import tensorflow as tf
from resnet18 import ModifiedResNet18

class SceneMemory(tf.keras.Model):

    def __init__(self, modalities=['image'], modality_dim={'image':64}, downsampleing_size=(64,64), 
      reduce_factor=4, observation_dim=128):
        super(SceneMemory, self).__init__()
        self.downsampling_size = downsampling_size
        self.modalities = modalities
        self.modality_dim = modality_dim
        self.observation_dim = observation_dim # the dimension of embedding for observations
        self.reduce_factor = reduce_factor
        self.embedding_nets = dict()
        self.memory = []

        self.downsampling = tf.keras.layers.AveragePooling2D(pool_size=(4,4),
          stride
          )

        if 'image' in modalities:
          self.embedding_nets['image'] = ModifiedResNet18(modality_dim['image'], reduce_factor)

        if 'pose' in modalities:
          self.embeddings_nets['pose'] = tf.keras.layers.Dense(modality_dim['pose'])

        if 'prev_action' in modalities:
          self.embeddings_nets['prev_action'] = tf.keras.layers.Dense(modality_dim['prev_action'])


        self.fc = tf.keras.layers.Dense(observation_dim)
    

    def call(self, observations, training=None):
      curr_embedding  = self._update(observations, training)

      return curr_embedding, tf.stack(self.memory, axis=1)


    def _update(self, observations, training=None):
      observations = tf.image.resize(observations,
        size=self.downsampling_size,
        method=ResizeMethod.BILINEAR)
    
      curr_embedding = self._embed(observations, training)
      self.memory.append(curr_embedding)

      return curr_embedding


    def _embed(self, observations, training=None):
       embeddings = []

       for modality in modalities.keys():
        if modality == 'image':
          embeddings.append(self.embedding_nets[modalities](observations[modality], training))
        else:
          embeddings.append(self.embedding_nets[modalities](observations[modality]))

       concat_embedding = tf.concat(embeddings, axis=1)

       return self.fc(concat_embedding)

