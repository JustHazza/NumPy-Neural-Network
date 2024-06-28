import numpy as np 

class NN_Model: 

    def __init__(self, layers): 
        self.layers = layers


class NN_FC_Layer:
    
    def __init__(self, num_neurons, W, name, eta = 1e-3, beta = 0, b= None): 

        #Adaptable Attributes upon construction
        self.num_neurons = num_neurons
        self.W = W
        self.b = np.zeros(num_neurons) if b is None else b
        self.name = name
        self.eta = eta
        self.beta = beta

        # Built in Attributes
        self.z = 0
        self.a = 0
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(b)
        self.delta = 0
        self.reg = 0
        self.grad = 0

    @staticmethod
    def ReLU(x): 
        '''
        ReLU activation function
        THE fastest vectorized implementation for ReLU
        '''

        x[x<0] = 0 
        return x

    def forward_prop(self, X):

        ''' 
        self: Fully Connected Neural Network Layer 
        X: Input into the layer: 
            - If from input layer, will be pixels from the image
            - Otherwise inputs from previous layer   
        '''
    
        # Apply Activation function (Only past input layer)
        if self.name == "Input": 
            self.a = X
        
        else: 
            self.a = self.ReLU(X)
        
        # Manipulate by Weights & Bias
        self.z = np.matmul(X, self.W) + self.b
    

    def back_propagation(self, X, previous_delta, previous_grad, next_z1 = None):

        '''
        self: Fully Connected Neural Network Layer
        X: Input into the layer: 
            - If from input layer, will be pixels from the image
            - Otherwise, inputs from previous layer (previous as in front previous,
            in terms of backprop its the next layer)

        previous_delta: Previous delta calculated from deeper layer 
            in backpropagation. ie. comes from layer 4 if this is layer 3

        previous_grad: previous grad calculated from deeper layer, in backprop
            ie. comes from layer 4 if this is layer 3.

        next_z1: Next z1 -> layer shallower than the current layer. ie 
                next_z1 comes from layer 2 if this is layer 3
        '''

        # Hidden Layer, n -> Hidden Layer n+1 weights' derivative
        # delta1 is \partial a2/partial z1
        # Hidden Layers activation's (weak) derivative is 1*(z1>0)

        if next_z1 is None: 
            self.grad = np.matmul(X.T, previous_delta)

        else:
            self.delta = np.matmul(previous_delta, self.W.T)*(next_z1>0)
            self.grad = np.matmul(X.T, self.delta)
       
        #Update Backpropagating Layers 
        N = int(self.z.shape[0])
        self.dW = (previous_grad / N) + ((self.reg / N) * self.W)
        self.db = np.mean(previous_delta, axis=0)


    def gradient_descent(self, GAMMA, EPS): 
        '''
        Gradient Descent: Updating the weights and biases alongside adding in 
        regularisation terms to the cost function allowing for learning of the model.

        Using RMSprop as the optimiser function -> there are others availble e.g Adam. 

        Inputs: 
        ETA, GAMMA, EPS: Hyper-Params defined for RMSprop.
        '''
        N = int(self.z.shape[0])

        gW = np.ones_like(self.W)
        gb = np.ones_like(self.b)

        #Update Weights
        gW = GAMMA*gW + (1 - GAMMA) * np.sum(self.dW**2)
        eta_W = self.eta/np.sqrt(gW + EPS)
        self.W -= eta_W * self.dW

        #Update Biases
        gb = GAMMA*gb + (1-GAMMA)*np.sum(self.db**2)
        eta_b = self.eta/np.sqrt(gb + EPS)            
        self.b -= eta_b * self.db

        #Random reset of gW, gB for improvement of gradient descent: 
        rand = np.random.rand()
        if rand < 0.02: 
            gW = np.ones_like(self.W)
            gb = np.ones_like(self.b)

    def glial_reg(self):

        '''
        Glial Regularisation: 

            - Makes the assumption that weights close to one another should theoretically be similiar
            and therefore penalises large differences in weights. Nearby weights will converge to 
            become more similar in order to lower cost function.

        '''

        #Transposing so that inputted weights can be reashaped in to original shape
        W0 = self.W.T

        #Setting shape of square reshaping
        m = int(np.sqrt(W0.shape[1]))

        #Reshaping Arrays (for each neuron)
        W0 = np.array(W0).reshape(-1, m, m)

        #Vectored across / down calculating element-wise distance squared.
        D_down = np.sum(np.sum((W0[:, :-1, :] - W0[:, 1:, :])**2, axis=(1, 2)))
        D_across = np.sum(np.sum((W0[:, :, :-1] - W0[:, :, 1:])**2, axis=(1, 2)))

        D = D_down + D_across

        self.reg = self.beta * D

    def L2_reg(self): 

        '''
        L2 Regularisation: Standard formula D = alpha (Weights ** 2)
        
        ''' 
        self.reg = np.sum(self.beta * np.sum(self.W**2, axis = 0))


class NN_Output_Layer: 

    def __init__(self, num_output, name = "Output"): 
    
        #Adaptable Attributes
        self.num_output = num_output
        self.name = name
        
        #Built in attributes
        self.y = 0
        self.y_encode = 0
        self.pred_values = np.zeros(num_output)
        self.delta = 0
        self.grad = 0
        

    def softmax(self, X, test = 0):
        '''
        Activation function for the last FC layer: softmax function 
        Output: K probabilities represent an estimate of P(y=k|X_in;weights) for k=1,...,K
        the weights has shape (n, K)
        X: Weights from previous FC layer
        '''

        # Subtract the maximum value for each row for normalisation stability.. need to catch 1 image error
        try:
            max_vals = np.max(X, axis=1, keepdims=True)
            s = np.exp(X - max_vals)
            total = np.sum(s, axis=1).reshape(-1,1)
            self.pred_values = s/total

        # Catches the 1 image error...
        except ValueError: 
            s = np.exp(X)
            self.pred_values = np.array(np.argmax(s))

        if test == 1: 
            return self.pred_values

    def one_hot_y(self): 
        '''
        One-hot encodes the y labels for predictions...
        '''
        self.y_encode = self.y[:,np.newaxis] == np.arange(self.num_output)

    def back_propagation(self, X): 

        '''
        self: The output layer of FC NN
        X: The input from previous layers weights
        '''
        
        self.delta = (self.pred_values - self.y_encode)
        self.grad = np.matmul(X.T, self.delta)

