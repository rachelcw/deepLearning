from constants import *
from data import *
from model import *
from model_corr import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--directory", help="directory of RBNS files", default="/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RBNS_training")
    args = parser.parse_args()
start_time=datetime.datetime.now()
print("start time: ", start_time)
directory = '/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RBNS_training'
RBNS_training_files = get_RBNS_files(directory) # all protein files

# PROTEIN_NAME = 'RBP1'
def workflow_single_protein_data(PROTEIN_NAME):
    RBNS_per_protein = get_files_per_protein(RBNS_training_files, PROTEIN_NAME) # what files to work on?
    # create master list of all sequences from the RBNS files and their labels
    master_list = create_master_list(RBNS_per_protein)
    class_lables = create_class_labels(RBNS_per_protein, PROTEIN_NAME)
    one_hot = convert_master_list_to_one_hot(master_list)
    RNAcompete_master_list_one_hot, num_seqs, shifts_RNAcompete_master_list = create_RNAcompete_master_list()
    print("start CNN: ", datetime.datetime.now())
    model = create_CNN_per_protein(one_hot, class_lables) # hyperparameters tunning
    print("end CNN: ", datetime.datetime.now())
    prediction, features = find_proteins_features(model, shifts_RNAcompete_master_list)
    return features

PROTEINS=['RBP1','RBP2','RBP3'] 
def workflow_fit_all_protein_data():
    # for all proteins
    # for protein in PROTEINS:
    #     features = workflow_single_protein_data(protein)
    # TODO: check shape and structure for both
    print("start corr model: ", datetime.datetime.now())
    all_features = [workflow_single_protein_data(protein) for protein in PROTEINS]
    all_y_target = [y_target_per_protein(protein) for protein in PROTEINS]
    simple_model, formula = create_simple_NN_model(all_features, all_y_target)
    print("end corr model: ", datetime.datetime.now())
    

# main
# if __name__ == "__main__":
#     directory=get_args()
#     RBNS_training_files = get_RBNS_files(directory) # all protein files

def print_cpu_usage():
    cpu_percentages = psutil.cpu_percent(percpu=True) # Get CPU percentage for each core
    overall_cpu_percentage = psutil.cpu_percent() # Get overall CPU percentage (average across all cores)
    print("CPU Usage per core:")
    for i, cpu_percentage in enumerate(cpu_percentages, start=1):
        print(f"Core {i}: {cpu_percentage:.2f}%")
    print("\nOverall CPU Usage:")
    print(f"{overall_cpu_percentage:.2f}%")

workflow_fit_all_protein_data()
end_time=datetime.datetime.now()
print("end time: ", end_time)
runtime = end_time - start_time
print("Total runtime:", runtime)
print_cpu_usage()