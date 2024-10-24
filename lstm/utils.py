import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.lib import xla_bridge
from matplotlib import pyplot as plt
import numpy as np

def split_and_discard(arrays):
    new_arrays = []
    lengths = [len(arr) for arr in arrays]
    print(lengths)
    MIN = np.min(lengths) 
    MAX = np.max(lengths)    

    print("minmax",MIN,MAX)

    for arr in arrays:
        if len(arr) == MIN:
            new_arrays.append(arr)
        elif len(arr) < MAX:
            new_arrays.append(arr[:MIN])
            new_arrays.append(arr[MIN:2*MIN])
        else:
            raise ValueError("Array size invalid")
    return new_arrays

#create arrays using the minimum length
def split_and_discard_sparse(arrays):
    new_arrays = []
    lengths = [len(arr) for arr in arrays]
    print(lengths)
    MIN = np.min(lengths) 
    MAX = np.max(lengths)    

    print("minmax",MIN,MAX)

    for arr in arrays:
        new_arrays.append(arr[:MIN])
        if len(arr)> 2*MIN:
            new_arrays.append(arr[MIN:2*MIN])
    return new_arrays

def normalize_sequence(seq):
    new_seq = (seq-np.min(seq)) / (np.max(seq)-np.min(seq))
    return new_seq

def values_above_threshold(input_list, threshold):
    above_threshold = [value for value in input_list if value > threshold]
    return above_threshold

def data_generator(arrays, batch_size=6):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
            if(end>dataset_size): print("end_epochs ",end)

def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
            if(end>dataset_size): print("end_epochs ",end)