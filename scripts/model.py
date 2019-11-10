# -*- coding: utf-8 -*-
"""

"""
# Imports
import numpy as np
import tensorflow as tf

from help import *

def prelu(I, name='prelu'):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    alphas = tf.get_variable('alpha', I.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                           dtype=tf.float32)
    tf.summary.histogram(name='alpha', values=alphas)

    pos = tf.nn.relu(I)
    neg = alphas * (I - tf.abs(I)) * 0.5

    O = pos + neg
  return O

def subsample(I, i_c, o_c, ks, stri, training=True, name='subsample'):
  # Get input shape
  shape = tf.shape(I)
  # kernel size
  k_h, k_w = ks

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    # Kernel initialization selection initialized to diagonal array    
    W = tf.get_variable(name='Weight', 
                        shape=[k_h, k_w, i_c, o_c], 
                        initializer=tf.orthogonal_initializer(), 
                        dtype=tf.float32, 
                        trainable=True)
    O = tf.nn.conv2d(I, 
                     W, 
                     strides=[1, stri[0], stri[1], 1], 
                     padding="SAME",
                     name='conv2d')
    # layer normalization
    O = tf.contrib.layers.layer_norm(O, reuse=tf.AUTO_REUSE, scope='layer_norm')
    O = prelu(O, name='prelu')
  return O

def upsample(I, i_c, o_c, ks, stri, training=True, last_layer=False, name='upsample'):
  # input shape
  shape = tf.shape(I)

  # output shape
  out_height = shape[1] * stri[0]    # time
  out_width = shape[2] * stri[1]     # frequency
  
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):   
    W = tf.get_variable(name='Weight', 
                        shape=[ks[0], ks[1], o_c, i_c],
                        initializer=tf.orthogonal_initializer(), 
                        dtype=tf.float32,
                        trainable=True)
    O = tf.nn.conv2d_transpose(I, 
                               W, 
                               output_shape=[shape[0], out_height, out_width, o_c], 
                               strides=[1, stri[0], stri[1], 1],
                               padding="SAME",
                               name='conv2d_transpose')
    if not last_layer:
      # layer normalization
      O = tf.contrib.layers.layer_norm(O, reuse=tf.AUTO_REUSE, scope='layer_norm')
      O = prelu(O, name='prelu')
  return O

def Analysis(signals, frame_length, frame_step):
  """
  signals: shape=[batch, height]
  return
    spec_real: tf.float32, shape=[batch, n_frames, fft_length]"""
  with tf.name_scope("Analysis"):
    # frame
    framed_signals = frame(signals, frame_length, frame_step)
    # x to xe
    xe = x2xe(framed_signals)
    # DFT
    spec = rdft(xe)
    # Take the real part of the spectrum, the imaginary part of the xe spectrum is approximately 0
    spec_real = tf.cast(spec, tf.float32)

  return spec_real

def Synthesis(spec, frame_length, frame_step):
  """
  spec: float32, shape=[batch, n_frames, fft_length]"""
  with tf.name_scope("Synthesis"):
    spec = tf.cast(spec, tf.complex64)
    # iDFT
    x_e = irdft(spec, N=2047)
    # xe to x
    x_framed = xe2x(x_e, frame_length)
    # Recovery signal
    signals = over_lap_and_add(x_framed, frame_length, frame_step)
  return signals

def getModel(I, training):
  """Unet-10"""
  res = list()
  # (128x1024x1)
  O = subsample(I, 1, 45, (5,7), (2,2), training, name='subsample_1')
  res.append(O)
  # (64x512x32)
  O = subsample(O, 45, 90, (5,7), (2,2), training, name='subsample_2')
  res.append(O)
  # (32x256x64)
  O = subsample(O, 90, 90, (3,5), (2,2), training, name='subsample_3')
  res.append(O)
  # (16x128x64)
  O = subsample(O, 90, 90, (3,5), (2,2), training, name='subsample_4')
  res.append(O)
  # (8x64x64)
  O = subsample(O, 90, 90, (3,5), (1,2), training, name='subsample_5')
  # (8x32x64)
  O = upsample(O, 90, 90, (3,5), (1,2), training, name='upsample_5')
  # (8x64x64)
  O = tf.concat([O, res.pop()], axis=-1, name='Concat_1')
  # (8x64x128)
  O = upsample(O, 180, 90, (3,5), (2,2), training, name='upsample_4')
  # (16x128x64)
  O = tf.concat([O, res.pop()], axis=-1, name='Concat_2')
  # (16x128x128)
  O = upsample(O, 180, 90, (3,5), (2,2), training, name='upsample_3')
  # (32x256x64)
  O = tf.concat([O, res.pop()], axis=-1, name='Concat_3')
  # (32x256x128)
  O = upsample(O, 180, 45, (5,7), (2,2), training, name='upsample_2')
  # (64x512x32)
  O = tf.concat([O, res.pop()], axis=-1, name='Concat_4')
  # (64x512x64)
  O = upsample(O, 90, 1, (5,7), (2,2), training, last_layer=True, name='upsample_1')
  # (128x1024x1)

  return O

def end_to_end(Input, is_training, d):
  frame_length = d.n_window
  frame_step = d.stride
  #
  # Stage 0，
  #
  spec = Analysis(Input, frame_length, frame_step)
  # 
  # Stage 1
  #
  with tf.variable_scope('Model'):
    # Increase channel dimension
    O = tf.expand_dims(spec, axis=-1)
    O = getModel(O, is_training)
    # Remove channel dimension
    O = tf.squeeze(O, axis=-1)
  #
  # Stage 2
  #
  with tf.name_scope('IRM'):
    irm = tf.math.tanh(0.5 * O) * 2.
    tf.summary.histogram(name='irm', values=irm)
    output_spec = irm * spec
    # the last dimension are all zeros which is the same as tensor "spec"
    #output_spec = output_spec[..., :-1]
  #
  # Stage 3
  #
  Output = Synthesis(output_spec, frame_length, frame_step)

  return Output
