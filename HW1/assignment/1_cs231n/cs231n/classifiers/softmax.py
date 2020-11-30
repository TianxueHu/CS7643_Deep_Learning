import numpy as np
from random import shuffle

def sftmax(z):
  numer = np.exp(z - np.max(z, axis=0))
  denom = np.sum(numer, axis=0) #sum of rows
  softmax = np.divide(numer, denom) # C x N
  return softmax

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C, D = W.shape
  N = X.shape[1]

  #Class score using softmax
  z = np.matmul(W, X) # z = Wx + b
  
  softmax = sftmax(z)

  #cross-entropy loss
  regu = reg*np.sum(W*W)
  #print("regu",regu)
  loss = -(1/N) * np.sum(np.log(softmax[y, range(N)] + 1e-5))  + regu
  #print(loss)

  softmax[y, range(N)] -= 1.0
  dW = (1/N) * np.dot(softmax, X.T)# C x N, N x D = C x D
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
