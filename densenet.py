import tensorflow as tf

class DenseNet:
  def __init__(self, features, mode):
    self.params = {
        'growth_rate':
        'dropout_rate':
        'k':
        'reduction':
        'bottleneck':
        }
    self.layers = [features]
    self.mode = mode_

  def dense_layer(self):
    layers = self.layers
    activation = tf.nn.relu # probably not going to change but just in case
    params = self.params

    layers.append(tf.layers.batch_normalization(layers[-1], training=self.mode))
    layers.append(activation(layers[-1]))
    if params['bottleneck']:
      layers.append(tf.layers.conv2d(
        inputs=layers[-1],
        filters=4*params['k'],
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        activation=activation,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
        bias_regularizer=tf.contrib.layers.l2_regularizer(1.0)))
      if params['dropout_rate'] > 0:
        layers.append(tf.nn.dropout(layers[-1], params['dropout_rate']))
      layers.append(tf.layers.batch_normalization(layers[-1],
        training=self.mode))
      layers.append(activation(layers[-1]))
    layers.append(tf.layers.conv2d(
      inputs=layers[-1],
      filters=params['k'],
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='same',
      activation=activation,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
      bias_regularizer=tf.contrib.layers.l2_regularizer(1.0)))
    if params['dropout_rate'] > 0:
      layers.append(tf.nn.dropout(layers[-1], params['dropout_rate']))

  def dense_block(self, layers_in_block):
    layers = self.layers
    activation = tf.nn.relu # probably not going to change but just in case
    params = self.params

    concat = layers[-1]
    for i in xrange(layers_in_block):
      dense_layer(layers, activation, params)
      concat = tf.concat([concat, layers[-1]], 3)
      layers.append(concat)

  def transition_block(self, last):
    layers = self.layers
    activation = tf.nn.relu # probably not going to change but just in case
    params = self.params

    layers.append(tf.layers.batch_normalization(layers[-1], training=self.mode))
    layers.append(activation(layers[-1]))
    if last:
      pass
      # average pool 7x7
      # reshape
    else:
      compressed = tf.floor(params['reduction'] * layers[-1].get_shape()[-1])
      layers.append(tf.layers.conv2d(
        inputs=layers[-1],
        filters=compressed,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        activation=activation,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
        bias_regularizer=tf.contrib.layers.l2_regularizer(1.0)))
      if params['dropout_rate'] > 0:
        layers.append(tf.nn.dropout(layers[-1], params['dropout_rate']))
      layers.append(tf.nn.pool(
        input=layers[-1],
        window_shape=[2, 2]
        pooing_type='avg',
        padding='same',
        strides=[2, 2]
        ))


