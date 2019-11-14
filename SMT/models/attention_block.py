import tensorflow as tf
import numpy as np
from attention import MultiHeadAttention

class AttentionBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(AttentionBlock, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ln1 = tf.keras.layers.LayerNormalization()
    self.ln2 = tf.keras.layers.LayerNormalization()
    self.dense = tf.keras.layers.Dense(d_model, activation=tf.nn.relu)

  def call(self, v, k, q, mask):
    att = self.mha(v, k, q, mask)
    h = self.ln1(tf.nn.relu(tf.keras.layers.add([att, q])))
    output =  self.ln2(tf.keras.layers.add(self.dense(h), h))

    return output