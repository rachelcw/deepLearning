from constants import *

"""create master list of all sequences from the RBNS files and their labels"""
def get_RBNS_files(directory):
    """read all RBNS files in directory and return a list of all files name in directory"""
    RBNS_training_files = []
    for filename in os.listdir(directory): # Iterate over all files in the directory
        if os.path.isfile(os.path.join(directory, filename)):
            RBNS_training_files.append(filename)
    return RBNS_training_files

def get_files_per_protein(RBNS_training_files, protein):
    """ filter RBNS files by single protein """
    # if len(protein) == 1:
        # list files in single protein - filter out protein 1 files only
    filtered_list = [value for value in RBNS_training_files if value.startswith(protein+'_')]
    # order file names according to concentration
    modified_list = [value.replace(f'{protein}_', '').replace('nM.seq', '') for value in filtered_list]
    if f'{protein}_input.seq' in filtered_list: # get rid of input so only ints are left
        modified_list.remove('input.seq')
    modified_list = sorted(modified_list, key=int)
    modified_list = modified_list[:2]+modified_list[-3:] # take only 3 lowest and 3 highest -- multiclass of 6
    modified_list = [protein+"_"+str(modified_list[i])+"nM.seq" for i in range(len(modified_list))]
    modified_list.insert(0,str(protein)+"_input.seq")
    # else:
    #     filtered_list = [value for value in RBNS_training_files for p in protein if value.startswith(p+'_')]
    return modified_list

def create_master_list(RBNS_per_protein):
    N = LIMIT_FILE_N_SEQ_READ * len(RBNS_per_protein) # number of seqs
    master_list = np.empty((N), dtype=f'|S{L_RBNS}') # sequences
    n=0
    for file in RBNS_per_protein:
    #print(file_index)
        file_path = '/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RBNS_training/' + file #TODO
        with open(file_path, 'r') as f:
            lines = [f.readline() for _ in range(LIMIT_FILE_N_SEQ_READ)] # read LIMIT_FILE_N_SEQ_READ lines
        # cut seq to 20 length randomly if len is longer find a random starting index and take 20 from there
        if file in FILES_40: # file that their seq len 40
            # choose random index to start from and select 20 chars - save shortend seq into master_list
            start_index = random.randint(0, 20)
            for line in lines:
                seq = line.split('\t')[0] # take only the sequence
                master_list[n] = seq[start_index : start_index + 20] # shortened seq
                n += 1
        else: # file len 20 + isnt input.seq
            for line in lines:
                seq = line.split('\t')[0] # take only the sequence
                master_list[n] = seq # seq
                n += 1
    del lines
    rand_master_list = np.empty(LIMIT_FILE_N_SEQ_READ , dtype=f'|S{L_RBNS}') # sequences
    ABC = [b'A', b'C', b'G', b'T']
    random.seed(613)
    rand_master_list = np.array([b''.join([ABC[random.randint(0,3)] for _ in range(L_RBNS)]) for _ in range(LIMIT_FILE_N_SEQ_READ)], dtype=f'|S{L_RBNS}')
    master_list = np.concatenate((master_list, rand_master_list))
    return master_list

def create_class_labels(RBNS_per_protein,protein):
    """create class labels for each sequence in master list"""
    N = LIMIT_FILE_N_SEQ_READ * len(RBNS_per_protein) # number of seqs
    class_lables = np.zeros((N, len(RBNS_per_protein)), dtype=np.int8) # array of probolities per class
    for i, file in enumerate(RBNS_per_protein): # set the lables for the rest of the classes
        if file != str(protein + '_input.seq'):
            class_lables[LIMIT_FILE_N_SEQ_READ * i: LIMIT_FILE_N_SEQ_READ * (i + 1), i:] = 1
    random.seed(613)
    rand_class_labels = np.zeros((LIMIT_FILE_N_SEQ_READ, len(RBNS_per_protein)), dtype=np.int8)
    class_lables = np.concatenate((class_lables, rand_class_labels))
    return class_lables

def convert_master_list_to_one_hot(master_list):
    one_hot = np.array([[ONE_HOT_DICT[bytes([nuc])] for nuc in (b"N" * PAD_SIZE + seq + b"N" * PAD_SIZE)] for seq in master_list])
    return one_hot
    
# cut seg from RNAcomplete file into all shifts of len 24
def create_RNAcompete_master_list():
    # read txt file
    file_path = '/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RNAcompete_sequences.txt'
    with open(file_path, 'r') as file:
        # convert seqences to one hot encoding + pad seq 
        RNAcompete_master_list_one_hot = [[ONE_HOT_DICT2[nuc] for nuc in ((41 - len(seq.rstrip('\n'))) * 'N' + seq.rstrip('\n'))] for seq in file if len(seq) > 5]
    num_seqs = len(RNAcompete_master_list_one_hot)
    # SHIFTS - RNAcomplete file into all shifts of len 24 -- create k-mers of all possible shifts
    step=5
    shifts_RNAcompete_master_list = [[seq[i:i+SEQ_PADDED_LEN] for i in range(0, len(seq) - SEQ_PADDED_LEN + 1, step)] for seq in RNAcompete_master_list_one_hot] # step +5 
    shifts_RNAcompete_master_list = np.reshape(shifts_RNAcompete_master_list, (-1, SEQ_PADDED_LEN, 4))
    return RNAcompete_master_list_one_hot, num_seqs, shifts_RNAcompete_master_list

def y_target_per_protein(protein):
    with open(f'/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RNCMPT_training/{protein}.txt','r') as file: #TODO
        y_target=np.array([float(line.strip()) for line in file if line != ""])
    return y_target