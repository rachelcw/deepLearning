# 1) load packages env py_dl hii
import os
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd 
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, regularization
from keras.layers import Embedding, Reshape, Activation
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import model_to_dot 
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.layers import SimpleRNN
from keras import initializers
from IPython.display import SVG
os.environ['OMP_NUM_THREADS'] = '50'
import datetime
start_time = datetime.datetime.now()

# 2) constants and hyperparameters
L_RBNS = 20 # length of each sequence in RBNS data
O = int(1e7) # for initializing big arrays, helps reduce runtime
LIMIT_FILE_N_SEQ_READ = int(1e6) # limit the amount of seq we read from file, helps reduce runtime
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

# 3) read files 
# list of files in RBNS_training
directory = '/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RBNS_training'
RBNS_training_files = []
for filename in os.listdir(directory): # Iterate over all files in the directory
    if os.path.isfile(os.path.join(directory, filename)):
        RBNS_training_files.append(filename)
print(RBNS_training_files)

# filter out protein 1 files only
PROTEIN = "RBP1" ### TODO change 
filtered_list = [value for value in RBNS_training_files if value.startswith(str(PROTEIN)+'_')]

prefixes = [item.split('_', 1)[0] for item in filtered_list]
suffixes = [item.split('_', 1)[1] for item in filtered_list]

# find the middle affinity files
modified_list = [value.replace('RBP1_', '').replace('nM.seq', '') for value in filtered_list]
if 'RBP1_input.seq' in filtered_list: # get rid of input
    modified_list.remove('input.seq')
modified_list = sorted(modified_list, key=int)
# modified_list = [PROTEIN + "_" + value + "nM.seq" for value in modified_list[1:3]]  # the first 2 files
modified_list = [PROTEIN + "_" + value + "nM.seq" for value in [modified_list[1], modified_list[-2]]]  # the first 2 files
modified_list.append(str(PROTEIN+'_input.seq')) # 0


# 4) BINARY classification - 
# initalize np.arrays
master_list = np.empty((O), dtype=f'|S{L_RBNS}') # sequences
class_lables = np.zeros((O, len(modified_list))) # array of probolities per class
n = 0

## running on protein 1 files only ############################################################
for file_index, file in enumerate(modified_list):
    #print(file_index)
    file_path = '/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RBNS_training/' + file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # choose sequences from file randomly - limited to LIMIT_FILE_N_SEQ_READ
        rng = np.random.default_rng(seed=123)
        rand_indices = rng.choice(len(lines), size=LIMIT_FILE_N_SEQ_READ, replace=False)

        # cut seq to 20 length randomly if len is longer find a random starting index and take 20 from there
        if file in FILES_40: # file len 40
            # choose random index to start from and select 20 chars - save shortend seq into master_list
            start_index = random.randint(0, 20)
            for index in rand_indices:
                seq = lines[index].split('\t')[0] # take only the sequence
                master_list[n] = seq[start_index:start_index+20] # shortened seq
                if file == str(PROTEIN+'_input.seq'):
                    class_lables[n, file_index] = 0
                else:
                    class_lables[n, file_index] = 1 # lables - 1 if in file otherwise stays 0
                n += 1
        elif file == str(PROTEIN+'_input.seq'):
              for index in rand_indices:
                    seq = lines[index].split('\t')[0] # take only the sequence
                    master_list[n] = seq # seq
                    class_lables[n, file_index] = 0 # lables - 1 if in file otherwise stays 0
                    n += 1
        else: # file len 20 + isnt input.seq
            for index in rand_indices:
                    seq = lines[index].split('\t')[0] # take only the sequence
                    master_list[n] = seq # seq
                    class_lables[n, file_index] = 1 # lables - 1 if in file otherwise stays 0
                    n += 1

# free memory initlized that wasnt used 
master_list = master_list[:n]
class_lables = class_lables[:n]
del lines

# binary vec
class_lables_binary = np.where(((class_lables[:, 0] == 1) | (class_lables[:, 1] == 1)) & (class_lables[:, 2] == 0), 1, 0)
class_lables_binary

# convert master_list seqences to one hot encoding
one_hot = np.zeros((len(master_list),  L_RBNS, 4), dtype=np.float16)
one_hot = np.array([ONE_HOT_DICT[bytes([nuc])] for seq in master_list for nuc in seq])
one_hot = one_hot.reshape(len(master_list), -1, 4)
print("done one hot-ing")

# 6) split into training and validation sets -- 80% training, 20% validation
# TODO LATER: keep out 6~ proteins for validation of the end
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
one_hot_train_X, one_hot_valid_X, one_hot_train_Y, one_hot_valid_Y = train_test_split(one_hot, class_lables_binary, test_size=0.2, random_state=42)
# train_X, test_X, train_y, test_y = train_test_split(one_hot, class_labels, test_size=0.2, random_state=42)
print("done train test split")

# load the model
model = keras.models.load_model("/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/model_sig_80_320.h5")

# cut seg from RNAcomplete file into all shifts of len 20
# read txt file
file_path = '/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RNAcompete_sequences.txt'
with open(file_path, 'r') as file:
    # convert seqences to one hot encoding
    RNAcompete_master_list_one_hot = [[ONE_HOT_DICT2[nuc] for nuc in ((41 - len(seq.rstrip('\n'))) * 'N' + seq.rstrip('\n'))] for seq in file if len(seq) > 5]
# cut seg from RNAcomplete file into all shifts of len 20 -- create k-mers of all possible shifts 
# shifts_RNAcompete_master_list = [[seq[i:i+20] for i in range(len(seq) - 20 + 1)] for seq in RNAcompete_master_list_one_hot]
shifts_RNAcompete_master_list = [[seq[i:i+20] for i in range(0, len(seq) - 20 + 1, 5)] for seq in RNAcompete_master_list_one_hot]
print("done shifts")
import concurrent.futures
import numpy as np
import multiprocessing
from functools import partial


# Assuming you have a Keras model named 'model' and a list 'shifts_RNAcompete_master_list'
# You may need to import your Keras model or create it here.

def get_max_prediction(shifts):
    return max(model.predict(shifts))

# Get the number of CPU cores available on your system
num_cores = 90 #multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# Using ThreadPoolExecutor to parallelize the predictions
print("starting pred")
with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    predictions = list(executor.map(get_max_prediction, shifts_RNAcompete_master_list))
end_time = datetime.datetime.now()

# Calculate and print the total runtime
runtime = end_time - start_time
print("Script started at:", start_time)
print("Script finished at:", end_time)
print("Total runtime:", runtime)
print("done pred")
predictions = np.array(predictions)  # Convert predictions list to a NumPy array if needed

with open("/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/predicition1_RBP1_all_80_320.txt", "w") as file:
    for pred in predictions:
        file.write(f'{pred}\n')

predictions

# print time stamps into txt file
with open("/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/time_stamps_RBP1_all_80_320.txt", "a") as file:
    file.write(f'Script started at: {start_time}\n')
    file.write(f'Script finished at: {end_time}\n')
    file.write(f'Total runtime: {runtime}\n')
               