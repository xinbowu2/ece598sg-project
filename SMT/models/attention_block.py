import tensorflow as tf
import numpy as np
from models.attention import MultiHeadAttention

class AttentionBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, epsilon=1e-6, rate=0.1):
    super(AttentionBlock, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.epsilon = epsilon
    self.rate = rate

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ln1 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
    self.ln2 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

    self.ffn = point_wise_feed_forward_network(d_model, d_model)

  def call(self, v, k, q, mask, training):
    '''
    att = self.mha(v, k, q, mask)
    h = self.ln1(tf.nn.relu(tf.keras.layers.add([att, q])))
    output =  self.ln2(tf.keras.layers.add(self.dense(h), h))
    '''
    att, att_weights = self.mha(v, k, q, mask)
    att = self.dropout1(att, training=training)
    h = self.ln1(tf.keras.layers.add([att, q]))

    ffn_output = self.ffn(h)
    ffn_output = self.dropout2(ffn_output, training=training)
    output = self.ln2(tf.keras.layers.add([ffn_output, h]))

    return output, att_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model, activation='relu')  # (batch_size, seq_len, d_model)
  ])
