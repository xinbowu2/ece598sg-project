import tensorflow as tf
import numpy as np
from attention_block import AttentionBlock

class AttentionPolicyNet(tf.keras.layers.Layer):
  def __init__(self, num_classes, d_model, num_heads=8, epsilon=1e-6, rate=0.1):
    super(AttentionPolicy, self).__init__()
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.d_model = d_model
    self.epsilon = epsilon
    self.rate = rate

    self.encoder = AttentionBlock(d_model, num_heads, epsilon, rate)
    self.decoder = AttentionBlock(d_model, num_heads, epsilon, rate)

    self.ffn = point_wise_feed_forward_network(d_model, num_classes) #??

  def call(self, o, m, mask, training):
    c = self.encoder(m, m, m, mask, training)
    decoder_output = self.decoder(c, c, o, mask, training)
    ffn_output = self.ffn(decoder_output)
    output = tf.keras.activations.softmax(x)
    # categorical distribution? 

    return output

def point_wise_feed_forward_network(dff, out_dim):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(out_dim)  # (batch_size, seq_len, d_model)
  ])