import urllib.request
import glob
import os
import numpy as np
from .HyperParameters import HP

def load_data():
  npzFile = np.load(HP.data_location, allow_pickle=True, encoding='latin1')
  train = npzFile['train']
  test = npzFile['test']
  valid = npzFile['valid']
  return train, valid, test

# Normalize input Dx, Dy. We only remove the std as explained in the paper
def calculate_normalizing_scale_factor(strokes):
    data = []
    for element in strokes:
        for point in element:
            data.append(point[0])
            data.append(point[1])
    return np.std(np.array(data))

def normalize(strokes):
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data
