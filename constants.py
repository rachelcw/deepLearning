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
Workers = 50
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
FILES_40 = ["RBP9" , "RBP11", "RBP15", "RBP31"] # files with sequences len 40 and not 20


# inputs --Hyperparameters
# NEED TO CHECK
NUM_KERNELS=128
EPHOCHS=3
BATCH_SIZE = 256
KERNEL_SIZE=3

STRIDES=1
STRIDES_BIG=3

# KEEP
FINAL_ACTIVATION_FUNCTION="sigmoid"
DROP_OUT=0.1
LAYERS=[64,32,32]
LR = 0.003

PAD_SIZE = KERNEL_SIZE - 1 # TODO: make sure this is correct to the selected final kernel size
SEQ_PADDED_LEN = 20 + 2 * PAD_SIZE

# STRIDES = 1
# STRIDES_BIG = 3
# DROP_OUT = 0.2 # helps with overfitting
# KERNEL_SIZE = [3, 5, 8]        
# NUM_KERNELS = [32, 128, 512]
# LAYERS = [128, 128] # ?
# FINAL_ACTIVATION_FUNCTION = "sigmoid"
# LR = [0.003, 0.01, 0.1] # convergence speed
# BATCH_SIZE = 256 # ?
# EPHOCHS = [3, 5, 7]
# POOL_SIZE = [None, 2, 4] # lowers time complexity
# POOL_TYPE = ["max", "avg"] # lowers time complexity


