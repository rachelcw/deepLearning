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
def create_simple_NN_model(features, y_target,protein):
        output_neurons = 1
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=(len(features[1]),))) # single neuron layer with 'linear' activation (no activation)

        # Compile model with the custom loss function
        model.compile(optimizer='adam', loss=pearson_correlation_loss, metrics=[pearson_correlation])
        # Fit the model to the data
        model.fit(features, y_target, batch_size=128, epochs=5, validation_split=0.5 ,multiprocessing=MultiProcess, workers=Workers)

        #### Create the formula for the Linear Regression model ####
        weights, biases = model.layers[0].get_weights()
        formula_parts = [f"{coeff}" for coeff in weights]
        formula = f"y = {biases[0]} (b) + " + " + ".join(formula_parts)
        print("NN Formula:", formula)
        ###########################################################

        prediction = model.predict(features, use_multiprocessing=MultiProcess, workers=Workers)
        with open(f'{protein}.txt', 'w') as f:
            for item in prediction:
                f.write("%s\n" % item)
        return model, formula