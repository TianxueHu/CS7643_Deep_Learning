import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  out = x.reshape([x.shape[0], -1]).dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  db = dout.sum(axis=0)
  dw = x.reshape([x.shape[0], -1]).T.dot(dout)
  dx = dout.dot(w.T).reshape(x.shape)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  out = np.maximum(0, x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = dout * (x > 0)
  return dx

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']

  #init
  out = np.zeros((N, F, int(1 + (H + 2 * pad - HH) / stride), int(1 + (W + 2 * pad - WW) / stride)), dtype=x.dtype)
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0)) # add padding to x

  #convolve
  for i in range(N):
      for j in range(F):
          for q in range(0, 1 + H + 2 * pad - HH, stride):
              for p in range(0, 1 + W + 2 * pad - WW, stride):
                    w_tmp = w[j,:,:,:]
                    x_tmp = x_pad[i, :, q:(q + HH), p:(p + WW)]
                    conv = np.multiply(w_tmp, x_tmp) 
                    cc  = np.sum(conv)
                    #update
                    out[i, j, int(q/stride), int(p/stride)] = cc + b[j] 


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']

  #init
  dx = np.zeros((N, C, H, W))
  dw = np.zeros((F, C, HH, WW))
  db = dout.sum(0).sum(1).sum(1) ##
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0)) # add padding to x
  dx_pad = np.zeros(x_pad.shape)

  #backward pass
  for i in range(N):
      for j in range(F):
          for q in range(0, 1 + H + 2 * pad - HH, stride):
              for p in range(0, 1 + W + 2 * pad - WW, stride):
                w_tmp = w[j,:,:,:]
                x_tmp = x_pad[i, :, q:(q + HH), p:(p + WW)]
                dw[j,:,:,:] += x_tmp * dout[i, j, int(q/stride), int(p/stride)]
                dx_pad[i, :, q:(q + HH), p:(p + WW)] += w_tmp * dout[i, j, int(q/stride), int(p/stride)]

  dx = dx_pad[:, :, pad:(pad + H), pad:(pad + W)]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  out = np.zeros((N, C, int(H/stride), int(W/stride)))

  for i in range(N):
      for j in range(C):
          for q in range(0, H - pool_height + 1, stride):
              for p in range(0, W - pool_width + 1, stride):
                x_tmp = x[i, j, q:(q+pool_height), p:(p+pool_width)]
                #update
                out[i, j, int(q/stride), int(p/stride)] = np.max(x_tmp) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  N, C, H, W = x.shape

  dx = np.zeros((N, C, H, W))

  for i in range(N):
      for j in range(C):
          for q in range(0, H - pool_height + 1, stride):
              for p in range(0, W - pool_width + 1, stride):
                x_tmp = x[i, j, q:(q+pool_height), p:(p+pool_width)]
                max_idx = np.argmax(x_tmp) # index of the max value
                idx_h, idx_w = np.unravel_index(max_idx, (pool_height, pool_width))
                dx[i, j, q+idx_h, p+idx_w] = dout[i, j, np.int(q/stride), np.int(p/stride)]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
