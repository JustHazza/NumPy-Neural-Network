import numpy as np 
import pandas as pd
from nn_numPy.nn_classes import NN_FC_Layer
from nn_numPy.nn_classes import NN_Output_Layer


def init_layers(N, NUM_NEURONS, ETA, BETA, K):

    '''
    Initialises the layers for the Neural Networks. 

    Inputs: 
    N: Integer: The number of pixels in the flattened input image
    NUM_NEURONS: nd.array: The number of Neurons in each layer 
    ALPHA: Float (double): Regularisation constant scalar
    K: Number of Class outputs

    Returns: 
    input_layer: A NN_FC_Layer object: Named as input
    hidden_layers: A list of NN_FC_Layer objects: Named sequentially 1, 2, 3 .. length of NUM_NEURONS
    output_layer: A NN_Output_Layer object 
    '''

    # Initialising Weights
    W_input = 1e-1 * np.random.randn(N, NUM_NEURONS[0])
    W_hidden = [1e-1 * np.random.randn(NUM_NEURONS[i], NUM_NEURONS[i+1]) for i in range(len(NUM_NEURONS) - 1)]
    W_hidden_final = 1e-1 * np.random.randn(NUM_NEURONS[-1], K)

    #Initilaising Bias
    b_input = 1e-1 * np.random.randn(NUM_NEURONS[0])
    b_hidden = [1e-1 * np.random.randn(NUM_NEURONS[i+1]) for i in range(len(NUM_NEURONS) - 1)]

    #Initialise names of hidden layers
    names = [i for i in range(1, len(NUM_NEURONS) + 1)]

    # Create FC layers + Output Layer
    input_layer = [NN_FC_Layer(num_neurons = N, W = W_input, b = b_input, name = "Input", eta = ETA, beta = BETA)]
    hidden_layers = [NN_FC_Layer(num_neurons=NUM_NEURONS, W=W, b =b, name = n, eta= ETA, beta = BETA) for NUM_NEURONS, W, b, n in zip(NUM_NEURONS, W_hidden, b_hidden, names)]
    output_layer = NN_Output_Layer(num_output=K, name = "Output")

    return input_layer, hidden_layers, output_layer


def org_data(file_path, train_split, perc = 10): 

    '''
    Reads in the CSV File - Splits into training & testing sets

    Inputs: 
    FILE_PATH: String: File path of the inputted csv file - currently set up for flattened images in csv and labels in column 1
    perc: Float (double): Percentage of Data from dataset used in training/test sets
    TRAIN_SPLIT: Float (double): In interval (0, 1) -> split ratio of train dataset ie. 0.8 = 80% train 20% test. 

    Returns: 

    X_test, X_train, y_test, y_train: nd.arrays of the split dataset  
    '''

    #Read in Data
    data = pd.read_csv(file_path)

    X = np.array(data)
  
    #Shuffle
    np.random.shuffle(X)
    m, n = X.shape
    
    #Apply Percentage
    X = X[0:int((perc/100)*m)]
    m, n = X.shape
    
    #Create Test/Train sets from X
    X_train = X[0: int(train_split*m):]
    X_test = X[int(train_split*m): m]

    #Pull out labels Y
    y_test = X_test[:, 0]
    y_train = X_train[:, 0]

    #Remove from X sets and normalise
    X_test = X_test[:, 1:n]/255.
    X_train = X_train[:, 1:n]/255.

    return X_test, X_train, y_test, y_train