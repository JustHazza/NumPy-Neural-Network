# NumPy-Neural-Network

This package is for educational purposes to aid the learning of the mathematics behind neural networks whilst also having fun creating my own neural networks from scratch which could then be used to attempt to implement a new form of regularisation. The neural network can be made with any number of layers, neurons and different types of regularisation although if you are looking for efficiency in training deep, deep neural networks I would stick to the well developed Tensorflow/PyTorch packages. It was created and used by Harry White in collaboration for his work with Joel Tabak at the University of Exeter, England. 

In order to create your neural network the nn_init.py file will give you all of the customisable parameters however, here is a brief example of how this can be done!

Firstly you'll need to import the modules...
```
#Self-Built neural network package
import nn_numPy.nn_classes as nn_layer
import nn_numPy.methods.nn_methods as nn_method
import nn_numPy.methods.nn_init as nn_init
```

Then set up parameters to use: 
```
#---------- Key Global Definitions/Params:  ------------#

#Random SEED
SEED = 120
np.random.seed(SEED)

# TRAIN_SPLIT Sets percentage of Train data
TRAIN_SPLIT = 0.8

# File Path for Dataset
FILE_PATH = "MNIST.csv" 

# Percentage of Dataset Used
PERC = 100

# Number of Classes
K = 10

# Number of Epochs
NUM_EPOCH = 400

#Number of Neurons
NUM_NEURONS = [196,64, K]

# Batch Size
BATCH_SIZE = 32

# Regularisation Setup
REGULARISATION = 'Glial'

#RMSprop HYPERPARAMS
GAMMA = 0.99 
EPS = 1e-8 

#  Bayesian Optimised 
BETA = 0.0067293
ETA = 0.0077402
```
Then create your dataset - say in this case we use the common MNIST.CSV for testing...

```
#Create Data set
X_test, X_train, y_test, y_train = nn_init.org_data(FILE_PATH, TRAIN_SPLIT, perc = PERC)

# Number of pixels in an image (Global Parameter)
N = X_train.shape[1]
```

Finally, you can initialise the layers of the network and train the model all in one handy function. 

```
#Create Model function: 
def create_model():
    #Initialise Layers (From Global Params Cell)
    input_layer, hidden_layers, output_layer = nn_init.init_layers(N, NUM_NEURONS, ETA, BETA, K)

    # List to hold all layers
    all_layers = input_layer + hidden_layers + [output_layer]

    # Train the Network!
    frames = nn_method.train(X_train, y_train, all_layers, 
                regularisation = REGULARISATION, 
                num_epochs = NUM_EPOCH, 
                batch_size = BATCH_SIZE,
                gamma = GAMMA, eps = EPS, 
                shuffle = True)

    return all_layers
```

Within this package there are various other methods used for testing accuracies but they should all be fully documented within their python files.
