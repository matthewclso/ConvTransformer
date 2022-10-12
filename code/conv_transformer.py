import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--train-samples", type = int, required = True, help = "how many samples of the training set to load")
parser.add_argument("--val-samples", type = int, required = True, help = "how many samples of the validation set to load")

parser.add_argument("--num-layers", type = int, required = True, help = "number of encoder-decoder layers in transformer")
parser.add_argument("--d-model", type = int, required = True, help = "number of units in transformer states")
parser.add_argument("--dff", type = int, required = True, help = "number of units in linear layers")
parser.add_argument("--num-heads", type = int, required = True, help = "number of heads in multi-head attention")
parser.add_argument("--dropout-rate", type = float, required = True, help = "proportion of weights to drop in dropout")
parser.add_argument("--kernels", nargs = "+", type = str, required = True, help = "the convolution filter size along the sequence dimension, 'd', the dilation rate - for example '5d3' is a 5-wide convolution with dilation rate 3")
parser.add_argument("--epochs", type = int, required = True, help = "maximum number of epochs during training")
parser.add_argument("--batch-size", type = int, required = True, help = "number of samples for each batch")
parser.add_argument("--target-acc", type = float, required = True, help = "accuracy in which to stop training")
parser.add_argument("--graph", type = bool, required = True, help = "whether or not to generate loss/accuracy graph")
parser.add_argument("--save-model", type = bool, required = True, help = "whether or not to save model")
args = parser.parse_args()

train_samples = args.train_samples
val_samples = args.val_samples

num_layers = args.num_layers
d_model = args.d_model
dff = args.dff
num_heads = args.num_heads
dropout_rate = args.dropout_rate
kernels = args.kernels
epochs = args.epochs
batch_size = args.batch_size
target_acc = args.target_acc
graph = args.graph
save_model = args.save_model

print(f"train_samples: {train_samples}")
print(f"val_samples: {val_samples}")

print(f"num_layers: {num_layers}")
print(f"d_model: {d_model}")
print(f"dff: {dff}")
print(f"num_heads: {num_heads}")
print(f"dropout_rate: {dropout_rate}")
print(f"kernels: {kernels}")

print(f"epochs: {epochs}")
print(f"batch_size: {batch_size}")
print(f"target_acc: {target_acc}")

# Limits for how many words can be considered
zh_vocab_size = 30000
en_vocab_size = 30000

# Limits for how long a sequence will be considered
zh_max_len = 200
en_max_len = 200

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import functools
import re

work_dir = os.getcwd()
os.chdir("/ibex/scratch/somc")

with h5py.File("zh_train_ds.hdf5", "r") as f:
  zh_train_ds = f["zh_train_ds"][:]
with h5py.File("en_train_ds.hdf5", "r") as f:
  en_train_ds = f["en_train_ds"][:]
with h5py.File("zh_val_ds.hdf5", "r") as f:
  zh_val_ds = f["zh_val_ds"][:]
with h5py.File("en_val_ds.hdf5", "r") as f:
  en_val_ds = f["en_val_ds"][:]
zh_timesteps, en_timesteps = zh_train_ds.shape[1], en_train_ds.shape[1]
training_samples, val_samples = zh_train_ds.shape[0], zh_val_ds.shape[0]
print(f"Chinese training dataset samples: {training_samples}   Chinese training dataset timesteps: {zh_timesteps}")
print(f"English training dataset samples: {training_samples}   English training dataset timesteps: {en_timesteps}")
print(f"Chinese validation dataset samples: {val_samples}   Chinese validation dataset timesteps: {zh_timesteps}")
print(f"English validation dataset samples: {val_samples}   English validation dataset timesteps: {en_timesteps}")

zh_tokenize = tf.keras.models.load_model("zh_tokenize")
en_tokenize = tf.keras.models.load_model("en_tokenize")

zh_train_ds = zh_train_ds[0:train_samples]
en_train_ds = en_train_ds[0:train_samples]
zh_val_ds = zh_val_ds[0:val_samples]
en_val_ds = en_val_ds[0:val_samples]

en_dec_train_ds = np.concatenate((en_train_ds[:, 1:], np.zeros((en_train_ds.shape[0], 1))), axis = 1)
en_dec_val_ds = np.concatenate((en_val_ds[:, 1:], np.zeros((en_val_ds.shape[0], 1))), axis = 1)

os.chdir(work_dir)

# Instead of using separate validation set, use samples from train set instead
zh_train_ds, zh_val_ds, en_train_ds, en_val_ds = train_test_split(zh_train_ds, en_train_ds, test_size = .03)

def positional_encoding(t, d_model):
  pos_encoding = np.array(np.arange(d_model))
  pos_encoding = 1 / np.power(10000, ((2 * (pos_encoding // 2)) / d_model))
  pos_encoding = [pos_encoding] * np.expand_dims(np.arange(t), 1)
  
  pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
  pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

  pos_encoding = np.expand_dims(pos_encoding, 0)

  return tf.cast(pos_encoding, dtype = tf.float32)

def padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :]

def future_mask(length):
  return 1 - tf.linalg.band_part(tf.ones((length, length)), -1, 0)

def sdp_attention(q, k, v, mask):
  qk = tf.matmul(q, k, transpose_b = True)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  qk /= tf.math.sqrt(dk)

  if mask is not None:
    qk += (mask * -1e9)

  softmax = tf.nn.softmax(qk, axis = -1)

  return tf.matmul(softmax, v)

class MaskedConv2D(tf.keras.layers.Layer):
  
  def __init__(self, filters, kernel_size, dilation_rate):
    super(MaskedConv2D, self).__init__()
    self.filters = filters
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.mask_width = self.kernel_size[0] // 2
    self.mask = tf.constant(tf.concat([tf.ones((self.mask_width + 1, self.kernel_size[1])),
                                       tf.zeros((self.mask_width, self.kernel_size[1]))], axis = 0))
    self.conv2d = tf.keras.layers.Conv2D(filters = filters,
                                         kernel_size = kernel_size,
                                         dilation_rate = dilation_rate)
    
  def build(self, input_shape):
    self.conv2d.build(input_shape)
    self.convolution_op = self.conv2d.convolution_op

  def masked_convolution_op(self, inputs, kernel, mask):
    return self.convolution_op(inputs, tf.math.multiply(kernel, tf.reshape(mask, self.mask.shape + [1, 1])))

  def call(self, x):
    self.conv2d.convolution_op = functools.partial(self.masked_convolution_op, mask = self.mask)
    return self.conv2d.call(x)

class EncoderMultiHeadAttention(tf.keras.layers.Layer):
  
  def __init__(self, d_model, num_heads, kernels):
    super(EncoderMultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.kernels = [kernel for kernel in kernels if bool(re.match("\d+d\d+", kernel))]
    
    self.filter_sizes = [int(kernel.split("d")[0]) for kernel in self.kernels]
    self.dilations = [int(kernel.split("d")[1]) for kernel in self.kernels]
    
    self.num_filters = [-(self.d_model // -len(self.filter_sizes))] * len(self.filter_sizes)
    self.num_filters[-1] = self.d_model - ((len(self.num_filters) - 1) * self.num_filters[0])
    
    for i in range(len(self.kernels)):
      if self.dilations[i] < 1:
        raise ValueError("dilation size must be 1 or more")
      if self.filter_sizes[i] % 2 != 1:
        raise ValueError("filter width must be an odd number")
      if self.filter_sizes[i] == 1 and self.dilations[i] != 1:
        raise ValueError("dilation is not valid if kernel width is 1")
    
    assert self.d_model % self.num_heads == 0

    self.dims = self.d_model // self.num_heads

    self.Wq = [tf.keras.layers.Conv2D(filters = self.num_filters[i],
                                      kernel_size = (self.filter_sizes[i], self.d_model),
                                      dilation_rate = (self.dilations[i], 1)) for i in range(len(self.kernels))]
    self.Wk = [tf.keras.layers.Conv2D(filters = self.num_filters[i],
                                      kernel_size = (self.filter_sizes[i], self.d_model),
                                      dilation_rate = (self.dilations[i], 1)) for i in range(len(self.kernels))]
    self.Wv = [tf.keras.layers.Conv2D(filters = self.num_filters[i],
                                      kernel_size = (self.filter_sizes[i], self.d_model),
                                      dilation_rate = (self.dilations[i], 1)) for i in range(len(self.kernels))]

    self.linear = tf.keras.layers.Dense(self.d_model)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
    return tf.transpose(x, perm = [0, 2, 1, 3])

  def call(self, qx, kx, vx, mask):
    batch_size = tf.shape(qx)[0]
    
    qs, ks, vs = [], [], []

    for i in range(len(self.filter_sizes)):
      pad_size = ((self.filter_sizes[i] - 1) * self.dilations[i]) // 2
      if pad_size > 0:
        qx_pad = tf.concat([tf.zeros((batch_size, pad_size, tf.shape(qx)[2])),
                            qx,
                            tf.zeros((batch_size, pad_size, tf.shape(qx)[2]))], axis = 1)
        kx_pad = tf.concat([tf.zeros((batch_size, pad_size, tf.shape(kx)[2])),
                            kx,
                            tf.zeros((batch_size, pad_size, tf.shape(kx)[2]))], axis = 1)
        vx_pad = tf.concat([tf.zeros((batch_size, pad_size, tf.shape(vx)[2])),
                            vx,
                            tf.zeros((batch_size, pad_size, tf.shape(vx)[2]))], axis = 1)
      else:
        qx_pad, kx_pad, vx_pad = qx, kx, vx
      qx_pad = tf.expand_dims(qx_pad, axis = -1)
      kx_pad = tf.expand_dims(kx_pad, axis = -1)
      vx_pad = tf.expand_dims(vx_pad, axis = -1)
      qi = self.Wq[i](qx_pad)
      ki = self.Wk[i](kx_pad)
      vi = self.Wv[i](vx_pad)
      qi = tf.squeeze(qi)
      ki = tf.squeeze(ki)
      vi = tf.squeeze(vi)
      qs.append(qi)
      ks.append(ki)
      vs.append(vi)
      
    q = tf.concat(qs, axis = 2)
    k = tf.concat(ks, axis = 2)
    v = tf.concat(vs, axis = 2)
    
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    sdp = sdp_attention(q, k, v, mask)

    sdp = tf.transpose(sdp, perm = [0, 2, 1, 3])

    sdp_concat = tf.reshape(sdp, (batch_size, -1, self.d_model))

    return self.linear(sdp_concat)
  
class DecoderMultiHeadAttention(tf.keras.layers.Layer):
  
  def __init__(self, d_model, num_heads, kernels):
    super(DecoderMultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.kernels = [kernel for kernel in kernels if bool(re.match("\d+d\d+", kernel))]
    
    self.filter_sizes = [int(kernel.split("d")[0]) for kernel in self.kernels]
    self.dilations = [int(kernel.split("d")[1]) for kernel in self.kernels]
    
    self.num_filters = [-(self.d_model // -len(self.filter_sizes))] * len(self.filter_sizes)
    self.num_filters[-1] = self.d_model - ((len(self.num_filters) - 1) * self.num_filters[0])
    
    for i in range(len(self.kernels)):
      if self.dilations[i] < 1:
        raise ValueError("dilation size must be 1 or more")
      if self.filter_sizes[i] % 2 != 1:
        raise ValueError("filter width must be an odd number")
      if self.filter_sizes[i] == 1 and self.dilations[i] != 1:
        raise ValueError("dilation is not valid if kernel width is 1")
    
    assert self.d_model % self.num_heads == 0

    self.dims = self.d_model // self.num_heads

    self.Wq = [MaskedConv2D(filters = self.num_filters[i],
                            kernel_size = (self.filter_sizes[i], self.d_model),
                            dilation_rate = (self.dilations[i], 1)) for i in range(len(self.kernels))]
    self.Wk = [MaskedConv2D(filters = self.num_filters[i],
                            kernel_size = (self.filter_sizes[i], self.d_model),
                            dilation_rate = (self.dilations[i], 1)) for i in range(len(self.kernels))]
    self.Wv = [MaskedConv2D(filters = self.num_filters[i],
                            kernel_size = (self.filter_sizes[i], self.d_model),
                            dilation_rate = (self.dilations[i], 1)) for i in range(len(self.kernels))]

    self.linear = tf.keras.layers.Dense(self.d_model)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
    return tf.transpose(x, perm = [0, 2, 1, 3])

  def call(self, qx, kx, vx, mask):
    batch_size = tf.shape(qx)[0]

    qs, ks, vs = [], [], []

    for i in range(len(self.filter_sizes)):
      pad_size = ((self.filter_sizes[i] - 1) * self.dilations[i]) // 2
      if pad_size > 0:
        qx_pad = tf.concat([tf.zeros((batch_size, pad_size, tf.shape(qx)[2])),
                            qx,
                            tf.zeros((batch_size, pad_size, tf.shape(qx)[2]))], axis = 1)
        kx_pad = tf.concat([tf.zeros((batch_size, pad_size, tf.shape(kx)[2])),
                            kx,
                            tf.zeros((batch_size, pad_size, tf.shape(kx)[2]))], axis = 1)
        vx_pad = tf.concat([tf.zeros((batch_size, pad_size, tf.shape(vx)[2])),
                            vx,
                            tf.zeros((batch_size, pad_size, tf.shape(vx)[2]))], axis = 1)
      else:
        qx_pad, kx_pad, vx_pad = qx, kx, vx
      qx_pad = tf.expand_dims(qx_pad, axis = -1)
      kx_pad = tf.expand_dims(kx_pad, axis = -1)
      vx_pad = tf.expand_dims(vx_pad, axis = -1)
      qi = self.Wq[i](qx_pad)
      ki = self.Wk[i](kx_pad)
      vi = self.Wv[i](vx_pad)
      qi = tf.squeeze(qi)
      ki = tf.squeeze(ki)
      vi = tf.squeeze(vi)
      qs.append(qi)
      ks.append(ki)
      vs.append(vi)
      
    q = tf.concat(qs, axis = 2)
    k = tf.concat(ks, axis = 2)
    v = tf.concat(vs, axis = 2)
    
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    sdp = sdp_attention(q, k, v, mask)

    sdp = tf.transpose(sdp, perm = [0, 2, 1, 3])

    sdp_concat = tf.reshape(sdp, (batch_size, -1, self.d_model))

    return self.linear(sdp_concat)
  
class EncoderUnit(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate = .1, kernels = ["1d1", "3d1", "5d1", "7d1"]):
    super(EncoderUnit, self).__init__()

    self.mha = EncoderMultiHeadAttention(d_model = d_model,
                                         num_heads = num_heads,
                                         kernels = kernels)
    
    self.ff1 = tf.keras.layers.Dense(dff, activation = "relu")
    self.ff2 = tf.keras.layers.Dense(d_model)

    self.norm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

    self.drop1 = tf.keras.layers.Dropout(rate)
    self.drop2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    mha_out = self.mha(x, x, x, mask)
    norm1 = self.drop1(mha_out, training = training)
    add1 = self.norm1(x + norm1)

    ff_out = self.ff1(add1)
    ff_out = self.ff2(ff_out)
    norm2 = self.drop2(ff_out, training = training)
    add2 = self.norm2(add1 + norm2)

    return add2

class Encoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, t, d_model, num_heads, dff, vocab_size, rate = .1, kernels = ["1d1", "3d1", "5d1", "7d1"]):
    super(Encoder, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.t = t

    self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_enc = positional_encoding(self.t, self.d_model)

    self.enc_units = [EncoderUnit(d_model = d_model,
                                  num_heads = num_heads,
                                  dff = dff,
                                  rate = rate,
                                  kernels = kernels) for _ in range(num_layers)]

    self.drop = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]
    x = self.embed(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_enc[:, :seq_len, :]

    x = self.drop(x, training = training)

    for enc in self.enc_units:
      x = enc(x, training, mask)

    return x
  
class DecoderUnit(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate = .1, kernels = ["1d1", "3d1", "5d1", "7d1"]):
    super(DecoderUnit, self).__init__()

    self.mha1 = DecoderMultiHeadAttention(d_model = d_model,
                                          num_heads = num_heads,
                                          kernels = kernels)
    self.mha2 = DecoderMultiHeadAttention(d_model = d_model,
                                          num_heads = num_heads,
                                          kernels = kernels)

    self.ff1 = tf.keras.layers.Dense(dff, activation = "relu")
    self.ff2 = tf.keras.layers.Dense(d_model)

    self.norm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
    self.norm3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

    self.drop1 = tf.keras.layers.Dropout(rate)
    self.drop2 = tf.keras.layers.Dropout(rate)
    self.drop3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_out, training, fut_mask, pad_mask):
    mha1_out = self.mha1(x, x, x, fut_mask)
    norm1 = self.drop1(mha1_out, training = training)
    add1 = self.norm1(x + norm1)

    mha2_out = self.mha2(add1, enc_out, enc_out, pad_mask)
    norm2 = self.drop2(mha2_out, training = training)
    add2 = self.norm2(add1 + norm2)

    ff_out = self.ff1(add2)
    ff_out = self.ff2(ff_out)
    norm3 = self.drop3(ff_out, training = training)
    add3 = self.norm3(add2 + norm3)

    return add3

class Decoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, t, d_model, num_heads, dff, vocab_size, rate = .1, kernels = ["1d1", "3d1", "5d1", "7d1"]):
    super(Decoder, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.t = t

    self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_enc = positional_encoding(self.t, self.d_model)

    self.dec_units = [DecoderUnit(d_model = d_model,
                                  num_heads = num_heads,
                                  dff = dff,
                                  rate = rate,
                                  kernels = kernels) for _ in range(num_layers)]

    self.drop = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_out, training, fut_mask, pad_mask):
    seq_len = tf.shape(x)[1]
    x = self.embed(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_enc[:, :seq_len, :]

    x = self.drop(x, training = training)

    for dec in self.dec_units:
      x = dec(x, enc_out, training, fut_mask, pad_mask)

    return x
  
class Transformer(tf.keras.Model):

  def __init__(self, num_layers, t, d_model, num_heads, dff, input_vocab_size, output_vocab_size, rate = .1, kernels = ["1d1", "3d1", "5d1", "7d1"]):
    super(Transformer, self).__init__()

    self.enc = Encoder(num_layers = num_layers,
                       t = t,
                       d_model = d_model,
                       num_heads = num_heads,
                       dff = dff,
                       vocab_size = input_vocab_size,
                       rate = rate,
                       kernels = kernels)
    self.dec = Decoder(num_layers = num_layers,
                       t = t,
                       d_model = d_model,
                       num_heads = num_heads,
                       dff = dff,
                       vocab_size = output_vocab_size,
                       rate = rate,
                       kernels = kernels)
    
    self.linear = tf.keras.layers.Dense(output_vocab_size)

  def call(self, inputs, training):
    x, y = inputs

    pad_mask = padding_mask(x)
    fut_mask = future_mask(tf.shape(y)[1])
    dec_pad_mask = padding_mask(y)
    fut_mask = tf.maximum(dec_pad_mask, fut_mask)

    enc_out = self.enc(x, training, pad_mask)

    dec_out = self.dec(y, enc_out, training, fut_mask, pad_mask)

    t_out = self.linear(dec_out)

    return(t_out)
  
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  
  def __init__(self, d_model, warmup_steps = 4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
  def get_config(self):
    config = {"d_model": self.d_model,
              "warmup_steps": self.warmup_steps}
    
    return config
  
# training on multiple GPUs with keras.model.fit
class accuracy_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if(logs.get("accuracy_function") >= target_acc):
      self.model.stop_training = True

acc_call = accuracy_callback()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = "none")

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss(real, pred)

  mask = tf.cast(mask, dtype = loss_.dtype)
  loss_ *= mask

  return tf.keras.backend.mean(tf.reduce_sum(loss_) / tf.reduce_sum(mask))

def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis = 2), tf.float32))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype = tf.float32)
  mask = tf.cast(mask, dtype = tf.float32)
  return tf.keras.backend.mean(tf.reduce_sum(accuracies) / tf.reduce_sum(mask))

en_train_ds_pad = np.concatenate((en_train_ds, np.zeros((en_train_ds.shape[0], 1))), axis = 1)
en_dec_train_ds = en_train_ds_pad[:, 1:]
en_train_ds = en_train_ds_pad[:, :-1]

en_val_ds_pad = np.concatenate((en_val_ds, np.zeros((en_val_ds.shape[0], 1))), axis = 1)
en_dec_val_ds = en_val_ds_pad[:, 1:]
en_val_ds = en_val_ds_pad[:, :-1]

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
  model = Transformer(
    num_layers = num_layers,
    t = zh_max_len,
    d_model = d_model,
    num_heads = num_heads,
    dff = dff,
    input_vocab_size = zh_vocab_size,
    output_vocab_size = en_vocab_size,
    rate = dropout_rate,
    kernels = kernels)

  lr = CustomSchedule(d_model)
  optimizer = tf.keras.optimizers.Adam(lr, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)
  model.compile(optimizer = optimizer, loss = loss_function, metrics = [accuracy_function])
  # history = model.fit([zh_train_ds, en_train_ds], en_dec_train_ds, batch_size = batch_size, epochs = epochs, callbacks = [acc_call])
  history = model.fit([zh_train_ds, en_train_ds], en_dec_train_ds, batch_size = batch_size, epochs = epochs, callbacks = [acc_call], validation_data = ([zh_val_ds, en_val_ds], en_dec_val_ds))
  
model.summary()

if graph:
  loss = history.history["loss"]
  accuracy = history.history["accuracy_function"]
  val_loss = history.history["val_loss"]
  val_accuracy = history.history["val_accuracy_function"]
  timerange = range(len(loss))

  fig,ax = plt.subplots()
  train_loss_plot, = ax.plot(timerange, loss, color = "blue")
  val_loss_plot, = ax.plot(timerange, val_loss, color = "cyan")
  train_loss_plot.set_label("Train Loss")
  val_loss_plot.set_label("Validation Loss")
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Loss")
  ax.legend(loc = "upper left")
  ax2 = ax.twinx()
  train_acc_plot, = ax2.plot(timerange, accuracy, color = "purple")
  val_acc_plot, = ax2.plot(timerange, val_accuracy, color = "pink")
  train_acc_plot.set_label("Train Accuracy")
  val_acc_plot.set_label("Validation Accuracy")
  ax2.set_ylabel("Accuracy")
  ax2.legend(loc = "upper right")
  plt.title("Loss vs Accuracy")
  plt.savefig(f"samples{train_samples}_dims{d_model}_date{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png")
  
if save_model:
  model.save(f"model_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
