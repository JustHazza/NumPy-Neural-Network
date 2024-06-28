import numpy as np
from nn_numPy.nn_classes import NN_FC_Layer
from nn_numPy.nn_classes import NN_Output_Layer

def describe(layers, N, K): 
    '''
    An Output Function to describe for each layer, what type of layer it is and the shapes of weights/biases (mostly for debugging purposes) 

    Input: 
    layers (list): A list of nn_classes objects (Fully Connected Layers or Output Layer)
    N: int: Number of pixels in original images
    K: Number of class outputs
    '''

    for layer in layers:
        if layer.name == "Input":
            print(f"The {layer.name} Layer with {N} inputs: Projected weights have shape {layer.W.shape}: Projected Biases have shape {layer.b.shape}")

        elif layer.name == "Output": 
            print(f"The {layer.name} Layer with {K} Outputs. No store of weights or biases")
        
        else: 
            print(f"Hidden Layer {layer.name} with {layer.num_neurons} neurons: Projected weights have shape {layer.W.shape}: Projected Biases have shape {layer.b.shape}")


def shuffle_data(X, y): 

    '''
    In place shuffle of the dataset - can be used in between each iteration for better learning

    X: Dataset (No Labels)
    y: Labels for dataset
    '''

    #Put Labels into X array
    X = np.concatenate((y.reshape(-1, 1), X), axis=1)
    
    #Shuffle
    np.random.shuffle(X)

    #Remove y labels again
    y = X[:, 0]

    #Remove from X
    X = X[:, 1:]

    return X, y


def forward_propagation(X, all_layers, test = 0):

    '''
    Runs the forward propogation for the designed neural network

    Inputs: 
    X: nd.array: Either the test/train set of the data. 
    all_layers: list: List of the layers being trained (FC or Output layers)
    test: Binary determinant of whether this is forward propogation for training or test. 
        test = 1: Running test propagation (ie. no update of weights)
        test != 1: Running training functions
        
    Returns (Only if Test = 1!): 
    ypreds: Array of predictions of y for the test set
    
    '''
    for layer in all_layers[:-1]:
        layer.forward_prop(X)

        #Output of the current layer is the input for next layer
        X = layer.z

    # If Using forward prop just to test network pass test = 1
    if test == 1: 
        y_preds = all_layers[-1].softmax(X, test = 1)
        return y_preds
    
    else:
        X = layer.z
        all_layers[-1].softmax(X)


def backward_propagation(X, y, all_layers):  

    '''
    Runs the back propagation of the designed neural network 

    Inputs: 
    X: nd.array: The train set of the data. 
    y: nd.array: The labels of the train set of the data
    all_layers: list: List of the layers being trained (FC or Output layers)

    '''
    #One Hot encode y here
    all_layers[-1].y = y
    all_layers[-1].one_hot_y()

    #Back Propagate starting with output -> HL
    all_layers[-1].back_propagation(X = all_layers[-2].a)

    #Loop Backwards through general number of hidden layers: HL_n+1 -> HL_n
    for i in range(len(all_layers)-2, 0, -1):
        all_layers[i].back_propagation(X = all_layers[i-1].a, 
                                    previous_delta = all_layers[i+1].delta, 
                                    previous_grad = all_layers[i+1].grad,
                                    next_z1 = all_layers[i-1].z)
        
    #Now Input Layer:
    all_layers[0].back_propagation(X = X, 
                                previous_delta = all_layers[1].delta, 
                                previous_grad = all_layers[1].grad,
                                next_z1 = None)
        

def glial_regularization(all_layers):
    '''
    Calculate glial regularization for each hidden layer

    Input: 
    all_layers: list: List of the layers being trained (FC or Output layers)
    '''
    for layer in all_layers[:-1]:
        layer.glial_reg()

def L2_regularization(all_layers):
    '''
    Calculate L2 regularization for each hidden layer

    Input: 
    all_layers: list: List of the layers being trained (FC or Output layers)
    '''
    for layer in all_layers[:-1]:
        layer.L2_reg()

def gradient_descent(all_layers, gamma, eps): 
    '''
    Perform gradient descent on all weights in layers

    Input: 
    all_layers: list: List of the layers being trained (FC or Output layers)
    '''
    for layer in all_layers[:-1]: 
        layer.gradient_descent(gamma, eps)

def loss_calc(y_pred, y, K):
    '''
    Loss function: cross entropy with an optional regularization
    y_pred: prediction made by the model, of shape (N, K) 
    y: ground truth, of shape (N, )
    N: number of samples in the batch
    '''

    # loss_sample stores the cross entropy for each sample in X
    # convert y_true from labels to one-hot-vector encoding

    #Add small constant to prevent log issues
    eps = 1e-10
    
    try:
        y_true_one_hot_vec = (y[:,np.newaxis] == np.arange(K))
        loss_sample = (np.log(y_pred + eps) * y_true_one_hot_vec).sum(axis=1)

    except IndexError: 
        print("Loss Function not required for single image")
        return 0

    # for the final loss, we need take the average

    return -np.mean(loss_sample)

def acc_calc(y_pred_final, y): 
    '''
    Performs Accuracy calculation and returns the value rounded to 3 dp. 

    Inputs: 
    y_pred_final: Predictions of y that the model has made
    y: The True values of y

    Returns: 
    acc: 
        - Calculates the mean of the value where the maximum value of the y predictions, matches
        the maximum value of y across the axis. Which will fall between 1 (if all correct or 0 if none correct). 
        The pure accuracy essentially. 
    
    Also catches error for 1 image!
    '''
    try:
        acc = np.round(np.mean(np.argmax(y_pred_final, axis=1) == y), 3)

    except np.AxisError: 
        acc = np.round(np.mean(np.argmax(y_pred_final) == y), 3)

    return acc
        

def test(X, y, all_layers, K): 

    '''
    Perform the testing of network 

    Inputs: 
    X: The test dataset for the network to learn off
    y: The labels of the training set
    all_layers: list: List of the layers being trained (FC or Output layers)
    K: Number of class labels

    Returns: 
    acc: accuracy of test
    loss: value of loss function
    '''
    
    #Forward Propagate with test flag
    y_pred_final = forward_propagation(X, all_layers, test = 1)
    loss = loss_calc(y_pred_final, y, K)
    acc = acc_calc(y_pred_final, y)
    
    return acc, loss


def visualise_weights(weights, num_epochs, neuron = None, timelapse = True): 

    import matplotlib.pyplot as plt
    
    '''
    Function to create a timelapse/video of the learning of the weights, saving it to the filepath.

    Inputs: 
    filepath: string of the file path required to save the video/timelapse
    fps: The fps of the video
    weights: the array of timestepped weights saved to be visualised
    neuron: If None, take 3 random neurons, else choose this/these neurons dependent on array/int. 
    timelapse: default: True - True will scale weights to be better for images, False will return original weights 
    '''
    #sqrt to find reshaped image size
    img_height = int(np.sqrt(np.shape(weights)[1]))

    #Check what neuron is first for efficiency of working on those
    # if neuron is None or (isinstance(neuron, np.ndarray) and neuron.size == 0)
    #     random_neuron = np.random.randint(low = 0, high = np.shape(weights)[2]) 
    #     neuron = np.array([random_neuron, random_neuron, random_neuron])

    neuron = np.array(neuron)

    frames_list = []

    #for every neuron chosen
    for n in neuron: 

        #reshape the weights
        W0 = np.transpose(weights, (0, 2, 1))
        W0 = np.reshape(W0[:, n], (-1, img_height, img_height))

        #create an empty image list
        frames = np.zeros([num_epochs + 1, img_height, img_height])
        
        #for every epoch
        for epoch in range(len(weights)):

            #create and add scaled image to the list
            if timelapse == True:
                image = np.uint8(255 * (W0[epoch] - np.min(W0[epoch])) / np.ptp(W0[epoch]))
                frames[epoch] = image

            else:
                frames[epoch] = W0[epoch]
        
        frames_list.append(frames)

        #save a video of the images put together for the passed filepath at fps given
    return frames_list




def visualise_train(X_train, y_train, all_layers, neuron = None, regularisation = 'None', 
                    num_epochs = 100, batch_size = 32, gamma = 0.99, eps = 1e-8, 
                    shuffle = True, timelapse = True):
    
    '''Visualise the training cycle by seeing the weight matrices evolve

    Inputs: 
    X_train: The training dataset for the network to learn off
    y_train: The labels of the training set
    all_layers: list: List of the layers being trained (FC or Output layers)
    neuron: the neuron / or list of neurons to visualise. 
    regularisation: Type of regularisation wanted can be 'None', 'L2' or 'Glial'
    num_epochs: Number of training cycles
    batch_size: Number of images per batch (ie before parameters updated!)
    GAMMA and EPS: Parameters for RMSprop
    shuffle: Default True: Shuffle the data in between each training epoch
    timelapse: Default True: Sets whether the images are scaled for better timelapses or not
    '''
    first_layer_weights = np.zeros([num_epochs + 1, np.shape(all_layers[0].W)[0], np.shape(all_layers[0].W)[1]])
    first_layer_weights[0] = all_layers[0].W

    #number of images in training set
    size = X_train.shape[0]

    #Run through Training Loop
    for epoch in range(num_epochs):

        #Split Into Batches: (Rest are discarded - hence shuffling every epoch)
        for i in range(int(size/batch_size)): 

            #Set the batches
            X = X_train[i*batch_size: i*batch_size + batch_size]
            y = y_train[i*batch_size: i*batch_size + batch_size]

            forward_propagation(X, all_layers, test = 0)
            backward_propagation(X, y, all_layers)

            # Regularisation choice : Glial, L2 or None
            if regularisation == 'Glial' or regularisation == 'glial':  
                glial_regularization(all_layers)

            elif regularisation == 'L2': 
                L2_regularization(all_layers)

            gradient_descent(all_layers, gamma, eps)
            first_layer_weights[epoch + 1] = all_layers[0].W

        #Conditional for Shuffling
        if shuffle == True: 
            X_train, y_train = shuffle_data(X_train, y_train)

    frames = visualise_weights(first_layer_weights, num_epochs, neuron, timelapse)
    return frames



def train(X_train, y_train, all_layers, regularisation = 'None', num_epochs= 100, batch_size = 32,
            gamma = 0.99, eps = 1e-8, shuffle = True, epochs_test = 0, test_set = None,
            test_labels = None, num_classes = None):

    '''
    Perform the training cycle 

    Inputs: 
    X_train: The training dataset for the network to learn off
    y_train: The labels of the training set
    all_layers: list: List of the layers being trained (FC or Output layers)
    regularisation: Type of regularisation wanted can be 'None', 'L2' or 'Glial'
    num_epochs: Number of training cycles
    batch_size: Number of images per batch (ie before parameters updated!)
    GAMMA and EPS: Parameters for RMSprop
    shuffle: Default True: Shuffle the data in between each training epoch

    epochs_test: Returns the test accuracy over every iteration (Used for reproduction of previous results)
    test_set, test_labels, num_classes: All used for parts of epochs_test: otherwise None. 
    '''
    
    #Create accuracies array for results reproduction
    if epochs_test == 1: 
        accuracies = np.zeros(num_epochs)

    #number of images in training set
    size = X_train.shape[0]

    #Run through Training Loop
    for epoch in range(num_epochs):

        #Split Into Batches: (Rest are discarded - hence shuffling every epoch)
        for i in range(int(size/batch_size)): 

            #Set the batches
            X = X_train[i*batch_size: i*batch_size + batch_size]
            y = y_train[i*batch_size: i*batch_size + batch_size]

            forward_propagation(X, all_layers, test = 0)
            backward_propagation(X, y, all_layers)

            # Regularisation choice : Glial, L2 or None
            if regularisation == 'Glial' or regularisation == "glial":  
                glial_regularization(all_layers)

            elif regularisation == 'L2': 
                L2_regularization(all_layers)

            gradient_descent(all_layers, gamma, eps)

        #Conditional for Shuffling
        if shuffle == True: 
            X_train, y_train = shuffle_data(X_train, y_train)

        # Conditional for epochs testing (Results Reproduction)
        if epochs_test == 1: 
            acc, _ = test(test_set, test_labels, all_layers, num_classes)
            accuracies[epoch] = acc


    if epochs_test == 1:
        return accuracies
            





    