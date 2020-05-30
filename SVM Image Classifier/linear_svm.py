import numpy as np

def svm_loss(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W

    Notation: 
    In convention, we have Wx where W is (C,D) and x is (D,1) matrices.
    However, since the label vector given is in row form, we take tranpose.
    Matrix-wise, 

    X = ---x_1---      W =   |    |    ...    |        s = ---s_1---
        ---x_2---            |    |    ...    |            ---s_2---
        .   .   .            w_1  w_2   ...   w_D          .   .   .
        ---x_N---            |    |    ...    |            ---s_N--- 
                             |    |    ...    |
             
    
    Each column of W corresponds to a category/class.

    Mathematical concepts to calculate gradient of multi-class SVM loss function
    For a single datapoint x_i, its loss function is
    L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]
    Differentiating with respect w_{y_i} (correct class) gives
    \nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i
    Differentiating with respect w_{j} where j \neq y_i (incorrect class) gives
    \nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i

    For all datapoints, we need to take average of all loss functions;
    L =  \frac{1}{N} \sum_{i=1}^N L_i +  \lambda R(W)
    In this function, we use l2-regularization, that is,
    R(W) = \sum_k\sum_l W_{k,l}^2
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    # s: A 2D array of shape (N, C) containing scores
    s = X.dot(W)

    # read correct scores into a column array of height N
    # recall that in array slicing, if lists are inputted, then slicing will 
    # return the elements at corresponding indices
    # for example, A = np.arange(9).reshape((3,3))
    # A[ [0,1,2], [0,1,2] ] will return diagonal elements [0,4,8]
    correct_score = s[list(range(num_train)), y] 
    correct_score = correct_score.reshape(num_train, -1)

    # subtract correct scores from score matrix and add margin
    # in other words, we calculate w_j^Tx_i - w_{y_i}^Tx_i + \Delta
    s = s - correct_score + 1   # delta = 1

    # make sure correct scores themselves don't contribute to loss function
    s[list(range(num_train)), y] = 0

    # construct loss function
    loss = np.sum(np.maximum(s, 0)) / num_train
    loss += reg * np.sum(W * W)

    # Implement a vectorized version of the gradient for the structured SVM
    # loss, storing the result in dW.                                     

    X_mask = np.zeros(s.shape)
    X_mask[s > 0] = 1
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
