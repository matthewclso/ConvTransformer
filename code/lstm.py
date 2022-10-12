import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--train-samples", type = int, required = True, help = "how many samples of the training set to load")
parser.add_argument("--val-samples", type = int, required = True, help = "how many samples of the validation set to load")

parser.add_argument("--units", type = int, required = True, help = "number of units in the LSTM state")
parser.add_argument("--embed-units", type = int, required = True, help = "number of units in the embedding layer")
parser.add_argument("--epochs", type = int, required = True, help = "maximum number of epochs during training")
parser.add_argument("--batch-size", type = int, required = True, help = "number of samples for each batch")
parser.add_argument("--target-acc", type = float, required = True, help = "accuracy in which to stop training")
args = parser.parse_args()

train_samples = args.train_samples
val_samples = args.val_samples

units = args.units
embed_units = args.embed_units
epochs = args.epochs
batch_size = args.batch_size
target_acc = args.target_acc

print(f"train_samples: {train_samples}")
print(f"val_samples: {val_samples}")

print(f"units: {units}")
print(f"embed_units: {embed_units}")
print(f"epochs: {epochs}")
print(f"batch_size: {batch_size}")
print(f"target_acc: {target_acc}")

# Limits for how many words can be considered
zh_vocab_size = 30000
en_vocab_size = 30000

# Limits for how long a sequence will be considered
zh_max_len = 200
en_max_len = 200

import tensorflow as tf
import numpy as np
import h5py
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt

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

class LSTMEncoderCell(tf.keras.layers.Layer):

  def __init__(self, units):
    super(LSTMEncoderCell, self).__init__()
    self.units = units
    self.state_size = (units, units)
    self.output_size = units

  def build(self, input_shape):
    self.w_update_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_update_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_update = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_forget_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_forget_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_forget = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_candidate_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_candidate_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_candidate = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_activation_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_activation_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_activation = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.built = True

  def call(self, inputs, states):
    prev_h = states[0]
    prev_c = states[1]

    update = tf.matmul(inputs, self.w_update_x)
    update += tf.matmul(prev_h, self.w_update_h)
    update += self.b_update
    update = tf.keras.activations.sigmoid(update)

    forget = tf.matmul(inputs, self.w_forget_x)
    forget += tf.matmul(prev_h, self.w_forget_h)
    forget += self.b_forget
    forget = tf.keras.activations.sigmoid(forget)

    candidate = tf.matmul(inputs, self.w_candidate_x)
    candidate += tf.matmul(prev_h, self.w_candidate_h)
    candidate += self.b_candidate
    candidate = tf.keras.activations.sigmoid(candidate)

    activation = tf.matmul(inputs, self.w_activation_x)
    activation += tf.matmul(prev_h, self.w_activation_h)
    activation += self.b_activation
    activation = tf.keras.activations.sigmoid(activation)

    c = tf.math.multiply(update, candidate) + tf.math.multiply(forget, prev_c)
    h = tf.math.multiply(activation, tf.keras.activations.sigmoid(c))
    
    return h, [h, c]

class LSTMManyInputDecoderCell(tf.keras.layers.Layer):

  def __init__(self, units, vocab_size):
    super(LSTMManyInputDecoderCell, self).__init__()
    self.units = units
    self.state_size = (units, units)
    self.output_size = units
    self.vocab_size = vocab_size

  def build(self, input_shape):
    self.w_update_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_update_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_update = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_forget_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_forget_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_forget = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_candidate_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_candidate_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_candidate = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_activation_x = self.add_weight(shape = (input_shape[-1], self.units), initializer = "random_normal", trainable = True)
    self.w_activation_h = self.add_weight(shape = (self.units, self.units), initializer = "random_normal", trainable = True)
    self.b_activation = self.add_weight(shape = (self.units,), initializer = "random_normal", trainable = True)

    self.w_softmax = self.add_weight(shape = (self.units, self.vocab_size), initializer = "random_normal", trainable = True)
    self.b_softmax = self.add_weight(shape = (self.vocab_size,), initializer = "random_normal", trainable = True)

    self.built = True

  def call(self, inputs, states):
    prev_h = states[0]
    prev_c = states[1]

    update = tf.matmul(inputs, self.w_update_x)
    update += tf.matmul(prev_h, self.w_update_h)
    update += self.b_update
    update = tf.keras.activations.sigmoid(update)

    forget = tf.matmul(inputs, self.w_forget_x)
    forget += tf.matmul(prev_h, self.w_forget_h)
    forget += self.b_forget
    forget = tf.keras.activations.sigmoid(forget)

    candidate = tf.matmul(inputs, self.w_candidate_x)
    candidate += tf.matmul(prev_h, self.w_candidate_h)
    candidate += self.b_candidate
    candidate = tf.keras.activations.sigmoid(candidate)

    activation = tf.matmul(inputs, self.w_activation_x)
    activation += tf.matmul(prev_h, self.w_activation_h)
    activation += self.b_activation
    activation = tf.keras.activations.sigmoid(activation)

    c = tf.math.multiply(update, candidate) + tf.math.multiply(forget, prev_c)
    h = tf.math.multiply(activation, tf.keras.activations.sigmoid(c))

    y = tf.matmul(h, self.w_softmax)
    y += self.b_softmax
    y = tf.keras.activations.softmax(y)
    
    return y, [h, c]
  
inputs_encoder = tf.keras.Input(shape = (zh_timesteps))
x = tf.keras.layers.Embedding(zh_vocab_size, embed_units, input_length = zh_timesteps, mask_zero = True)(inputs_encoder)
h, h_s, c_s = tf.keras.layers.RNN(LSTMEncoderCell(units), return_state = True)(x)

inputs_decoder = tf.keras.Input(shape = (en_timesteps))
h = tf.keras.layers.Embedding(en_vocab_size, embed_units, input_length = en_timesteps, mask_zero = True)(inputs_decoder)
y = tf.keras.layers.RNN(LSTMManyInputDecoderCell(units, en_vocab_size), return_sequences = True)(h, initial_state = [h_s, c_s])
model = tf.keras.Model(inputs = [inputs_encoder, inputs_decoder], outputs = y)
model.summary()

class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = "masked_loss"
    self.reduction = tf.keras.losses.Reduction.NONE
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = self.reduction)

  def __call__(self, y_true, y_pred, sample_weight):
    loss = self.loss(y_true, y_pred)

    mask = tf.cast(y_true != 0, tf.float32)
    loss *= mask

    return tf.reduce_sum(loss)

class accuracy_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if(logs.get("accuracy") >= target_acc):
      self.model.stop_training = True

acc_call = accuracy_callback()

# training on multiple GPUs
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
  inputs_encoder = tf.keras.Input(shape = (zh_timesteps))
  x = tf.keras.layers.Embedding(zh_vocab_size, embed_units, input_length = zh_timesteps, mask_zero = True)(inputs_encoder)
  h, h_s, c_s = tf.keras.layers.RNN(LSTMEncoderCell(units), return_state = True)(x)

  inputs_decoder = tf.keras.Input(shape = (en_timesteps))
  h = tf.keras.layers.Embedding(en_vocab_size, embed_units, input_length = en_timesteps, mask_zero = True)(inputs_decoder)
  y = tf.keras.layers.RNN(LSTMManyInputDecoderCell(units, en_vocab_size), return_sequences = True)(h, initial_state = [h_s, c_s])
  model = tf.keras.Model(inputs = [inputs_encoder, inputs_decoder], outputs = y)

  optimizer = tf.keras.optimizers.Adam(learning_rate = .01)
  loss = MaskedLoss()
  model.compile(optimizer = optimizer, loss = loss, metrics = ["accuracy"])
  history = model.fit([zh_train_ds, en_train_ds], en_dec_train_ds, batch_size = batch_size, epochs = epochs, callbacks = [acc_call])
  # history = model.fit([zh_train_ds, en_train_ds], en_train_ds, batch_size = 64, epochs = 50, validation_data = ([zh_val_ds, en_val_ds], en_dec_val_ds))
  
loss = history.history["loss"]
accuracy = history.history["accuracy"]
# val_loss = history.history["val_loss"]
# val_accuracy = history.history["val_accuracy"]
timerange = range(len(loss))

fig,ax = plt.subplots()
loss_plot, = ax.plot(timerange, loss, color = "blue")
loss_plot.set_label("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(loc = "upper left")
ax2 = ax.twinx()
acc_plot, = ax2.plot(timerange, accuracy, color = "purple")
acc_plot.set_label("Accuracy")
ax2.set_ylabel("Accuracy")
ax2.legend(loc = "upper right")
plt.title("Loss vs Accuracy")
plt.savefig(datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png")