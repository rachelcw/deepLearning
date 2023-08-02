from constants import *
from data import *

# Define the custom Pearson correlation coefficient loss function for model
def pearson_correlation_loss(y_true, y_pred):
    return 1 - pearson_correlation(y_true, y_pred)  # Return 1 - correlation to minimize the negative correlation

# Define the custom Pearson correlation coefficient function for model metric
def pearson_correlation(y_true, y_pred):
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    covariance = tf.reduce_mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_std = tf.math.reduce_std(y_pred)
    correlation = covariance / (y_true_std * y_pred_std)
    return correlation  # Return 1 - correlation to minimize the negative correlation

# Create a Sequential model with 1 output neurons for Pearson correlation
def create_simple_NN_model(features, y_target, protein):
        output_neurons = 1
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=(len(features[1]),))) # single neuron layer with 'linear' activation (no activation)

        # Compile model with the custom loss function
        model.compile(optimizer='adam', loss=pearson_correlation_loss, metrics=[pearson_correlation])
        # Fit the model to the data
        model.fit(features, y_target, batch_size=128, epochs=8, validation_split=0.3, use_multiprocessing=MultiProcess, workers=Workers, verbose=2)
        # save model to file
        model.save(f'corr_model_y.h5')
        # #### Create the formula for the Linear Regression model ####
        # weights, biases = model.layers[0].get_weights()
        # formula_parts = [f"{coeff}" for coeff in weights]
        # formula = f"y = {biases[0]} (b) + " + " + ".join(formula_parts)
        # print("NN Formula:", formula)
        return model #, formula


# get output files per protein with the predicted affinity score
def get_final_output_per_protein(features, protein_name):
    # read model file
    model = keras.models.load_model(f'corr_model_y.h5', custom_objects={"pearson_correlation_loss":pearson_correlation_loss, "pearson_correlation":pearson_correlation})
    prediction = model.predict(features, use_multiprocessing=MultiProcess, workers=Workers, verbose=2)
    with open(f'{protein_name}.txt', 'w') as f:
            for item in prediction:
                f.write("%s\n" % item)
    # return prediction

# check our correlation ststus after running full workflow
def get_corr(protein_name):
    with open(f'/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/RNCMPT_training/{protein_name}.txt','r') as file:
        y_target=[float(line.strip()) for line in file if line != ""]
    with open(f'{protein_name}.txt','r') as file:
         pred = [float(line.strip('[]\n')) for line in file if line != ""]
    
    correlation_coefficient, _ = stats.pearsonr(y_target, pred)
    print("corr is:", correlation_coefficient)