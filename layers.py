import tensorflow as tf
import numpy as np

####################################################
# Operational Layers.
class Oper2D(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation = None, q = 1, padding = 'valid', use_bias=True, strides=1):
    super(Oper2D, self).__init__(name='')

    self.activation = activation
    self.q = q
    self.all_layers = []

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                    (kernel_size,
                                                      kernel_size),
                                                    padding=padding,
                                                    use_bias=use_bias,
                                                    strides = strides,
                                                    activation=None))

  @tf.function
  def call(self, input_tensor, training=False):
    
    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))
    
    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x

####################################################
# Transposed Operational Layers.
class Oper2DTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation = None, q = 1, padding = 'valid', use_bias=True, strides=1):
    super(Oper2DTranspose, self).__init__(name='')

    self.activation = activation
    self.q = q
    self.all_layers = []

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                    kernel_size,
                                                    padding=padding,
                                                    use_bias=use_bias,
                                                    strides = strides,
                                                    activation=None))

  @tf.function
  def call(self, input_tensor, training=False):

    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x