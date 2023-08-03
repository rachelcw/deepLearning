from constants import *
from data import *

# 5) NN per protien - input each seq (len 20) in master_list into NN 
    # output = vec of probabilities (one per each concentration)
    # compare output to bool numpy array (true lable)
    # backpropogation to model to minimize loss?


# Define the metric for model
def accuracy_th(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast labels to float32
    threshold = 0.5
    y_pred_thresholded = tf.where(y_pred >= threshold, 1., 0.)  # Apply the threshold to the predicted probabilities
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_thresholded), tf.float32))  # Calculate the accuracy
    return accuracy

# create a CNN model per protein. input is one_hot seqs and output is prediction of probability of 6 class_lables
def create_CNN_per_protein(X,Y,model_name):
    # shuffle data
    rng = np.random.default_rng(32)
    CUT_DATA = int(2e6)
    print(f'X shape is: {X.shape}')
    print(f'Y shape is: {Y.shape}')
    shuf_inds = rng.choice(X.shape[0], size=CUT_DATA, replace=False)
    print(f'shuf_inds shape is: {shuf_inds.shape} ', max(shuf_inds))
    X = X[shuf_inds]
    Y = Y[shuf_inds]

    model = Sequential()
    ############## Yaron's model #######################
    model.add(Conv1D(filters=512, kernel_size=8, strides=1,
                   kernel_initializer='RandomNormal',
                   activation='relu',
                   input_shape=X.shape[1:], use_bias=True,
                   bias_initializer='RandomNormal'))
    model.add(MaxPooling1D(pool_size=5, strides=None,
                               padding='valid',
                               data_format='channels_last'))
    model.add(Flatten())
    for layer_size in LAYERS:
        print(f'the layer size is: {layer_size}')
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))

    Adam = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)
    model.compile(optimizer=Adam, loss='binary_crossentropy',metrics=[accuracy_th])
    ############################## our model ########################################
    # model.add(Conv1D(filters=NUM_KERNELS, kernel_size=KERNEL_SIZE[0], strides=STRIDES,
    #                 kernel_initializer=initializers.RandomNormal(stddev=0.01), activation='relu',
    #                 input_shape=X.shape[1:], use_bias=True, bias_initializer='RandomNormal'))
    
    # model.add(Conv1D(filters=NUM_KERNELS, kernel_size=KERNEL_SIZE[1], strides=STRIDES_BIG))
    # # model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=None, padding='valid', data_format='channels_last')) # MAYBE USE CONV KERNEL HERE INSTEAD 
    # model.add(Flatten())
    # model.add(Dropout(DROP_OUT))
    # # dense layers
    # for layer_size in LAYERS:
    #             print(f'the layer size is: {layer_size}')
    #             model.add(Dense(layer_size, activation='relu'))
    #             model.add(Dropout(DROP_OUT))
    # # output layer
    # model.add(Dense(6, activation=FINAL_ACTIVATION_FUNCTION)) 
    # model.compile(optimizer=Adam(learning_rate = LR), loss="binary_crossentropy", metrics=[accuracy_th])
    model.summary()
    ##########################################
    # Compile the model with Optimizer, Loss Function and Metrics
    run_hist_1 = model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPHOCHS, shuffle=True, use_multiprocessing=MultiProcess, workers=Workers, verbose=2)
    model.save(f'/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/{model_name}.h5') # save the model
    model = keras.models.load_model(f'/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/{model_name}.h5', custom_objects={"accuracy_th":accuracy_th})
    return model

# create features: MULTICLASS parallelize the prediction + output -  suppose predict gives N dim vector:
def find_proteins_features(model, shifts_RNAcompete_master_list,protein):
    N = 6 # number of classes
    prediction = model.predict(shifts_RNAcompete_master_list, use_multiprocessing=MultiProcess, workers=Workers, verbose=2)
    _,num_seqs,_ = create_RNAcompete_master_list()
    prediction = np.reshape(prediction, (num_seqs, -1, N))
    # fetures - 6 max + 6 min per seq with best shift
    features = np.empty((num_seqs, 2 * N))
    features[:, :N] = np.max(prediction, axis=1) # shape is (num_seqs, N)
    features[:, N:] = np.min(prediction, axis=1) # shape is (num_seqs, N)

    # # write pred into file
    # protein="RBP1"
    with open(f'/data01/private/resources/RACHELI_EDEN_SHARED/DL_PROJ/features_predicitions_{protein}.txt', "w") as file:
        for feature in features:
            file.write(f"{feature}\n")
    return prediction, features
