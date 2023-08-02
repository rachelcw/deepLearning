# 1) load packages env py_dl
import os
import argparse
import psutil
# os.environ['OMP_NUM_THREADS'] = '50'
import numpy as np
import pandas as pd 
import random
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from keras import initializers
from keras.optimizers import Adam #, SGD, RMSprop
import datetime
import scipy.stats as stats


# 2) constants and hyperparameters
Workers = 80
MultiProcess = True
L_RBNS = 20 # length of each sequence in RBNS data
O = int(1e7) # for initializing big arrays, helps reduce runtime
# LIMIT_FILE_N_SEQ_READ = 1000 # limit the amount of seq we read from file, helps reduce runtime
LIMIT_FILE_N_SEQ_READ = int(1e6/2) # limit the amount of seq we read from file, helps reduce runtime
ONE_HOT_DICT = {b'A': np.array([0,0,0,1], dtype=np.float16),
                b'C': np.array([0.,0.,1,0], dtype=np.float16),
                b'G': np.array([0,1,0,0], dtype=np.float16),
                b'T': np.array([1,0,0,0], dtype=np.float16),
                b'U': np.array([1,0,0,0], dtype=np.float16),
                b'N': np.array([0.25,0.25,0.25,0.25])
                }

ONE_HOT_DICT2 = {'A':[0,0,0,1],
                'C': [0,0,1,0],
                'G':[0,1,0,0],
                'T': [1,0,0,0],
                'U': [1,0,0,0],
                'N': [0.25,0.25,0.25,0.25]
                }

L_RBNS = 20 # length of each sequence in RBNS data
FILES_40 = ["RBP9" , "RBP11", "RBP15", "RBP31"] # files  with sequences len 40 and not 20
model_param_dict = {"kernel_size":3, "pool_size":2, "layers": [128, 128], "final_activation_function":"sigmoid"}
PAD_SIZE = model_param_dict["kernel_size"] - 1
SEQ_PADDED_LEN = 20 + 2 * PAD_SIZE