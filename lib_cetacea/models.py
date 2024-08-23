import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from lib_cetacea.patches import *
from lib_cetacea.attention_layers import *
from lib_cetacea.masking import * 


def encoder(length, dim, n_layers, n_heads):
  i = Input((length, dim))
  x = LayerNormalization()(i)
  for _ in range(n_layers):
    x = EncoderLayer(
      d_model = dim,
      num_heads = n_heads,
      dff = dim
    )(x)
  o = x
  return Model(inputs = [i], outputs = [o])


def sequence_classifier(enc, length, dim, patch_size, output_dim, label_sequence=False):
  i = Input((length, dim, 1))
  i = Patches(patch_size=patch_size)(i)
  e = enc(i)
  if not label_sequence:
    e = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(e)
  o = FeedForward(patch_size * patch_size, output_dim, activation='softmax')(e)
  return Model(inputs = [i], outputs = [o])


