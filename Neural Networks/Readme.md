# Two-layers Neural Network
We implemeneted a two-layer fully-connected neural network with one hidden layer from scratch so that I can understand its underlying 
working mechanism. 
We also plotted the underlying weights images to understand what the neurons in hidden layer do to input images. 

<p align="center"> <img  src="https://github.com/hongwai1920/Machine-Learning-algorithms/blob/master/Neural%20Networks/Images/hidden%20neurons.png" ></p> 


## two_layer_neural_net.ipynb
Contains implementation and training a two-layer neural networks on CIFAR-10 image dataset.
We also include tuning hypterparamaters (learning rate, regularization strength and number of neurons in the hidden layer) to 
achieve better performance on test set. 


## neural_net.py 
Contains a class `TwoLayerNet` 
  * A two-layer fully-connected neural network (one hidden layer). 
  * The net has an input dimension of N, a hidden layer dimension of H, and performs classification over C classes.
  * We train the network with a softmax loss function and L2 regularization on the weight matrices.  The network uses a ReLU nonlinearity after the first fully connected layer. 
  In other words, the network has the following architecture:
  * input - fully connected layer - ReLU - fully connected layer - softmax
  * The outputs of the second fully-connected layer are the scores for each class.

The class contains the following methods.
1. `__init__(self, input_size, hidden_size, output_size, std=1e-4)`,
2. `loss(self, X, y=None, reg=0.0)`,
3. `train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=5e-6, num_iters=100, batch_size=200, verbose=False)`,
4. `predict(self, X)`

## vis_utils.py
Contains methods for visualizing weights images.
1. `visualize_grid(Xs, ubound=255.0, padding=1)`,
2. `vis_grid(Xs)`,
3. `vis_nn(rows)`
