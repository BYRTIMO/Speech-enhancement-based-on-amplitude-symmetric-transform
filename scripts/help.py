# -*- coding: utf-8 -*-
"""
Script function: sequence framing, windowing, 
transformation of any sequence to conjugate symmetric sequence, 
dft and idft transformation, overlap addition, etc.
modify：
  1、dft Support unilateral spectrum generation, idft supports input spectrum as unilateral spectrum
"""
# Imports
import numpy as np
import tensorflow as tf

def frame(signals, frame_length, frame_step, winfunc=tf.signal.hamming_window):
  """Use the frame function in tf.signal to achieve framing
  Return：
    framed_signals: shape=[batch, n_frames, frame_length]"""
  framed_signals = tf.signal.frame(signals, frame_length, frame_step, pad_end=False)
  if winfunc is not None:
    window = winfunc(frame_length, dtype=tf.float32)
    framed_signals *= window
  return framed_signals

def x2xe(framed_signals):
  """Transform each frame signal of x to obtain its corresponding conjugate symmetric sequence, 
     flip first, then splicing
  """
  shape = framed_signals.shape
  # In addition to 0 points, the framing signal is halved
  frame0 = framed_signals[..., :1]
  frames_other = framed_signals[..., 1:] * 0.5
  framed_half = tf.concat([frame0, frames_other], axis=-1)
  # Generate flip tensor
  converted_frames = tf.reverse(framed_half, axis=[-1])
  # After splicing the inverted signal to the framing signal
  framed_xe = tf.concat([framed_half, converted_frames], axis=-1)
  # Remove the last dimension
  xe_framed = framed_xe[..., :-1]

  xe_framed = tf.cast(xe_framed, tf.float32)

  return xe_framed

def rdft(signals, N=2047, one_sided=True):
  """Realize the DFT transformation of the real sequence, support batch, 
     return only the first half, do not return the symmetric part
  Params： 
    signals: shape=[batch, n_frames, frame_length]
    N: Indicates the number of DFT points, which is generally equal to the length 
       of the frame. When N is even and only returns a one-sided sequence, the length of 
       the return sequence is N / 2 + 1; when N is an odd number, the return sequence length is (N + 1) /
    one_sided: Return to a one-sided sequence when True, return complete sequence when False
  return：
    complex_spec: tf.complex64，Complex spectrum
  """
  shape = tf.shape(signals)
  # Generate timeline subscript
  n_index = tf.reshape(np.arange(N), [1, N])
  n_index = tf.cast(n_index, tf.float32)
  n_index = tf.tile(n_index, [N, 1])
  # Generate frequency axis subscript
  k_index = tf.transpose(n_index)
  # There is no pi in tensorflow
  np_pi = np.pi
  pi = tf.constant(np_pi, dtype=tf.float32)

  # Generating Fourier Transform Matrix
  real_kernel = tf.math.cos(1. / N * (2 * pi) * (k_index * n_index))
  imag_kernel = tf.math.sin(1. / N * (2 * pi) * (k_index * n_index))
  # First expand the batch dimension
  sign = tf.reshape(signals, [-1, N])
  # Fourier transform
  real = tf.matmul(sign, real_kernel)
  imag = tf.matmul(sign, imag_kernel)
  # Generating complex spectrum
  complex_spec = tf.dtypes.complex(real=real, imag=imag) 
  complex_spec = tf.reshape(complex_spec, shape)
  # Whether to return a single-sided sequence
  if one_sided is True:
    # NWhen N is an odd number
    complex_spec = complex_spec[..., : (N + 1) // 2]

  return complex_spec

def irdft(spec, N=2047, one_sided=True):
  """Perform an inverse dft transform on the conjugate symmetric spectrum 
     to obtain a real sequence. When one_sided is equal to True, the complete spectrum is first restored.
  Params：
    spec: tf.complex64, shape=[batch, n_frames, fft_length], When spec is a single-sided sequence，fft_length=N // 2 + 1
    N: Indicates the number of DFT points. The default is equal to the 
       length of the real signal per frame. If N is an odd number, the value must be given.
    one_sided: When True, the input spectrum spec is a one-sided spectrum.
  Return：
    signals：tf.float32, shape=[batch, n_frames, frame_length]"""
  
  # Now only consider the case where N is odd
  if one_sided == True and N % 2 == 1:
    # Conjugate spectrum flip
    spec_reverse = tf.reverse(spec[..., 1: ], axis=[-1])
    spec = tf.concat([spec, spec_reverse], axis=-1)
  shape = tf.shape(spec)
 
  # Generate timeline subscript
  n_index = tf.reshape(np.arange(N), [1, N])
  n_index = tf.cast(n_index, tf.float32)
  n_index = tf.tile(n_index, [N, 1])
  # Generate frequency axis subscript, Transposes last two dimensions of tensor n_index
  k_index = tf.transpose(n_index)
  # There is no pi in tensorflow
  np_pi = np.pi
  pi = tf.constant(np_pi, dtype=tf.float32)
  # Generating Fourier Transform Matrix
  real_kernel = tf.math.cos(1. / N * (2 * pi) * (k_index * n_index))
  imag_kernel = -1 * tf.math.sin(1. / N * (2 * pi) * (k_index * n_index))
  # This piece combines the real part and the imaginary part, 
  # and directly calculates the complex matrix.
  # Generate inverse Fourier transform matrix
  inverse_matrix = 1. / N * tf.dtypes.complex(real=real_kernel, imag=imag_kernel)
  # First expand the batch dimension
  sign = tf.reshape(spec, [-1, shape[-1]])
  # Inverse Fourier transform
  signals = tf.matmul(sign, inverse_matrix)
  # In case of casting from complex types (`complex64`, `complex128`) to real
  # types, only the real part of `x` is returned. In case of casting from real
  # types to complex types (`complex64`, `complex128`), the imaginary part of the
  # returned value is set to `0`. The handling of complex types here matches the
  # behavior of numpy.
  signals = tf.cast(signals, tf.float32)
  signals = tf.reshape(signals, shape)

  return signals

def xe2x(framed_signals, frame_length):
  """Convert from xe signal to x signal
  Params：
    framed_signals: tf.float32, shape=[batch, n_frames, fft_length]
    frame_length: The length of the frame representing the signal x
  Return：
    x: tf.float32, shape=[batch, n_frames, frame_length], Indicates the original signal recovered"""
  # Take out the first dimension
  frame0 = framed_signals[..., :1]
  # Take out other dimensions and multiply by 2
  frames_other = framed_signals[..., 1:] * 2.
  # splice
  ori_signals = tf.concat([frame0, frames_other], axis=-1)
  # Take the front frame_length dimension and return
  x = ori_signals[..., : frame_length]
  return x

def over_lap_and_add(framed_signals, frame_length, frame_step, winfunc=tf.signal.hamming_window):
  """overlap and add
  params：
    framed_signals: tf.float32, shape=[batch, n_frames, frame_length]
    frame_length: Window length
    frame_step: frame shift
  return：
    signals: tf.float32, shape=[batch, x_length]
  """
  shape = tf.shape(framed_signals)
  n_frames = shape[1]
  # Generate de-overlapping windows
  if winfunc is not None:
    window = winfunc(frame_length, dtype=tf.float32)
    window = tf.reshape(window, [1, frame_length])
    window = tf.tile(window, [n_frames, 1])
    window = tf.signal.overlap_and_add(window, frame_step)
  signals = tf.signal.overlap_and_add(framed_signals, frame_step)
  signals /= window
  signals = tf.cast(signals, tf.float32)
  return signals