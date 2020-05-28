import numpy as np

class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        
        dists = self.compute_distances(X)

        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # Mathematical concepts: 
        # Compute (i,j)-th entry of dists
        # Let A = test[i] and B = train[j] 

        # # Note that both A and B are row vectors

        # By nearest neighbor definition, we have
        # [dist]_{i,j}^2
        # = \| A - B \|_2^2
        # = (A-B) (A-B)^T
        # = (A-B)(A^T -B^T)
        # = A A^T - A B^T - B A^T + B B^T
        # = A A^T - 2 A B^T + B B^T   (since A B^T is a scalar)
        # = np.sum(np.square(A)) - 2*[np.matmul(test, train.T)]_{i,j}  \ 
        #                        + np.sum(np.square(B))

        # Therefore, in general, 
        # dists^2 
        # = ( \|test[i] - train[j] \|_2^2 )_{i,j}
        # = ( np.sum(np.square(test[i])) - 2*( np.matmul(test, train.T) )_{i,j}  + np.sum(np.square(train[j]) ) )_{i,j}
        # = np.sum(np.square(test[i])) - 2*( np.matmul(test, train.T) )  + np.sum(np.square(train[j]) )

        # if reshape(-1,1) is not applied on first term, Python throws error:
        # ValueError: operands could not be broadcast together with shapes (500,) (500,5000) 

        dists = np.sqrt(np.sum(np.square(X), axis = 1).reshape(-1,1) - 2* np.matmul(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1))

        return dists


    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
          
            # np.argsort(dists[i])[:k] finds indexes with the first k smallest values 
            # in dists[i]
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred
