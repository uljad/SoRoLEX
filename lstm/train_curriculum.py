import math
import sys
import warnings
from typing import Tuple, TypeVar

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.lib import xla_bridge
from matplotlib import pyplot as plt
from utils import normalize_sequence

print(xla_bridge.get_backend().platform)
import csv
import functools
import os
import pickle
from datetime import date, datetime
from itertools import chain

import numpy as np
import optax
from flax import linen as nn
from flax.metrics import tensorboard

DATA = "data"
WINDOW_SIZE = 512                   
STEP = 1
batch_size = 64 # Batch size for training.
FEATURES_DIM = 1024


class RandomSequenceLoader():
    def __init__(self, data_directory):
        self.directory = data_directory

    def get_all_columns(self,csv_file):
        try:
            with open(csv_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)  # Get the header row
                data = [row for row in reader]

            # Transpose the data to get an array of arrays, where each array is a column
            data = np.array(data, dtype=float).T

            # Create a dictionary to store the columns with their corresponding names
            all_columns = {header[i]: data[i] for i in range(len(header))}
            return all_columns
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def getAllRawData(self,search_term):
        # List files that contain "NoDownsampling" in their name
        matching_files = [filename for filename in os.listdir(self.directory) if search_term in filename]
        raw_data=[] #list of dictionaries from all files
        print(matching_files)
        # Print the matching file names
        for filename in matching_files:
            # Extract all columns as an array of arrays
            cols = self.get_all_columns(os.path.join(self.directory, filename))
            # Print all columns
            raw_data.append(cols)
        return raw_data

def sliding_window(sequence, window_size, step_size):
    """
    Splits a sequence into overlapping sliding windows.

    Args:
    sequence (list or str): The input sequence.
    window_size (int): The size of the sliding window.
    step_size (int): The step size for moving the window.

    Returns:
    list: A list of arrays, where each array represents a sliding window.
    """
    windows = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        window = sequence[i:i + window_size]
        windows.append(window)
    return windows


search_term = "Step_cont"
raw_data=[] #list of dictionaries from all files
data_loader = RandomSequenceLoader(DATA)
raw_data = data_loader.getAllRawData(search_term)
print(len(raw_data))

long_positions = []
long_pressures = []
for r in raw_data:
    long_positions.append(np.array([r["1_x"],r["1_y"],r["1_z"]]).T)
    long_pressures.append(np.array([r["control_pressure_0"],r["control_pressure_1"],r["control_pressure_2"]]).T)

print(long_positions[0].shape)
print(long_pressures[1].shape)
print(len(long_positions))
print(len(long_pressures[0]))
for pr,pos in zip(long_pressures,long_positions):
    print(pr.shape)
    print(pos.shape)


pressure_train = []
pressure_test = []
position_train = []
position_test = []

test_index = [8,4,9]

pressure_test = [long_pressures[i] for i in test_index]
pressure_train = [long_pressures[i] for i in np.arange(len(long_pressures)) if i not in test_index]
print(len(pressure_test),len(pressure_train))

position_test = [long_positions[i] for i in test_index]
position_train = [long_positions[i] for i in np.arange(len(long_positions)) if i not in test_index]
print(len(pressure_test),len(pressure_train))
print(len(pressure_test[0]))


x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []

for pressure in pressure_train:
    x_train_list.append(np.array(sliding_window(pressure,WINDOW_SIZE,STEP)))

for pressure in pressure_test:
    x_test_list.append(np.array(sliding_window(pressure,WINDOW_SIZE,STEP)))

for pos in position_train:
    y_train_list.append(np.array(sliding_window(pos,WINDOW_SIZE,STEP)))

for pos in position_test:
    y_test_list.append(np.array(sliding_window(pos,WINDOW_SIZE,STEP)))


N_train = np.sum([len (i) for i in x_train_list])  # Replace N1, N2, N3 with your actual values
print("N_Train =================================", N_train)

# Create an empty array for the combined data
x_train = np.empty((N_train, WINDOW_SIZE, 3))
y_train = np.empty((N_train, WINDOW_SIZE, 3))

start_index = 0
for press,pos in zip(x_train_list,y_train_list):
    end_index = start_index + press.shape[0]
    x_train[start_index:end_index] = press
    y_train[start_index:end_index] = pos
    start_index = end_index

N_test =np.sum([len (i) for i in x_test_list])  # Replace N1, N2, N3 with your actual values
print("N_Test ==============================",N_test)

# Create an empty array for the combined data
x_test = np.empty((N_test, WINDOW_SIZE, 3))
y_test = np.empty((N_test,WINDOW_SIZE, 3))

start_index = 0
for press,pos in zip(x_test_list,y_test_list):
    end_index = start_index + press.shape[0]
    x_test[start_index:end_index] = press
    y_test[start_index:end_index] = pos
    start_index = end_index


class ForwardLSTM(nn.Module):
  features: int 
  in_carry: Tuple = None

  def setup(self):
      self.lstm_layer = nn.scan(
      nn.OptimizedLSTMCell,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})(self.features)

      self.lstm1 = self.lstm_layer
      self.dense1 = nn.Dense(3)
      self.layer_norm_1 = nn.LayerNorm()
      self.layer_norm_2 = nn.LayerNorm()
      
  @nn.remat    
  def __call__(self, x_batch):
      x = x_batch
      key1, key2 = random.split(random.PRNGKey(0))
      carry = self.lstm_layer.initialize_carry(key2,x[:, 0].shape)  
      x = self.lstm1(carry, x)[1] 
      x = self.layer_norm_1(x)
      x = nn.relu(x)
      x = self.dense1(x)
      return x
  
x = x_train[0:1]
x = jnp.ones((batch_size, WINDOW_SIZE, 3))
init_rngs = {'params': random.PRNGKey(0)}
model = ForwardLSTM(features=FEATURES_DIM)
params = model.init(init_rngs, x)
y = model.apply(params,x)

def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = indices.copy()
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
            if(end>dataset_size): print("end_epochs ",end)


train_iter = dataloader((x_train,y_train),batch_size=batch_size)
test_iter = dataloader((x_test,y_test),batch_size=batch_size)

@jax.jit
def l2_loss_func(params, x, y):
    pred_y = model.apply(params, x)
    return jnp.mean(optax.l2_loss(pred_y, y))

summary_writer_1 = tensorboard.SummaryWriter("jax_log/lstm/")
summary_writer_2 = tensorboard.SummaryWriter("jax_log/lstm/")
loss_grad_fn = jax.value_and_grad(l2_loss_func)

UPDATES = int(1)
tx = optax.adam(learning_rate=1e-3)
opt_state = tx.init(params)

for step, (x, y), (xs,ys) in zip(range(UPDATES),train_iter,test_iter):
  loss_val, grads = loss_grad_fn(params, x, y)
  updates, opt_state = tx.update(grads, opt_state)#adding params for adamw
  params = optax.apply_updates(params, updates)
  eval_loss, _ = loss_grad_fn(params, xs, ys)
  summary_writer_1.scalar('loss', loss_val, step)
  summary_writer_2.scalar('validations_loss', eval_loss, step)
  print(step,loss_val)


now = datetime.now()
today = date.today()
d1 = today.strftime("%b-%d-%Y ")
current_time = now.strftime("%H:%M")
checkpoint_name = os.path.join("checkpoints",d1 + current_time + ".pkl")
with open(os.path.join("checkpoints",d1 + current_time + ".pkl"),"wb")as out_file:
    pickle.dump(params, out_file)

print(checkpoint_name)
file =  open(os.path.join(checkpoint_name),"rb")
data = pickle.load(file)
params = data


TEST_INDEX = 12100
x = x_test[TEST_INDEX:TEST_INDEX+1,:,:]
print(x.shape)
y_pred = model.apply(params, x)
print(y_pred.shape)
print(x[0,:,0].shape)
plt.plot(y_test[TEST_INDEX,:,0:1],label="test_output")
plt.plot(y_pred[0,:,0:1],label="prediction")
plt.legend()
plt.savefig("lstm_prediction_slide_sigmoid_walk_"+str(TEST_INDEX)+".png")
