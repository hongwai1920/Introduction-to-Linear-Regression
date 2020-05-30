# SVM Image Classifier

## SVM_Image_Classifier.ipynb
This notebook implements a multi-class SVM Image classifier on the CIFAR-10 image dataset.

## linear_classifier.py

## linear_svm.py
Contains a function `svm_loss(W, X, y, reg)` to calculate SVM loss function.
For each image <img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" />, we flatten it into a 1D array.
Then the loss function we used is 
<p align="center"> <img  src="https://latex.codecogs.com/svg.latex?L_i&space;=&space;\sum_{j\neq&space;y_i}&space;\left[&space;\max(0,&space;w_j^Tx_i&space;-&space;w_{y_i}^Tx_i&space;&plus;&space;\Delta)&space;\right]" title="L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]"></p> 
where <img src="https://latex.codecogs.com/svg.latex?\Delta" title="\Delta" /> is a hypterparamter.
