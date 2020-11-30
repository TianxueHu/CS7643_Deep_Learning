---
layout:     page
title:      Homework 1, Programming Question
permalink:  /AP6c2L5Z4nNd9i4sB86r/hw1-coding/
---

# [CS 4803-7643 Deep Learning - Homework 1][5]

In this homework, we will learn how to implement backpropagation (or backprop) for 
“vanilla” neural networks (or Multi-Layer Perceptrons). You will begin by writing the forward and
backward passes for different types of layers, and then go on to train a neural network on the CIFAR-10 dataset in Python. Next you’ll learn to use [PyTorch][3], a popular open-source deep learning framework,
and use it to replicate the experiments from before.

This homework is divided into the following parts:

- Implement a neural network and train it on CIFAR-10 in Python.
- Learn to use PyTorch and replicate previous experiments in PyTorch (2-layer NN on CIFAR-10).

Download the starter code [here]({{site.baseurl}}/assets/f20cs7643_hw1_starter.zip).

## Part 1

Starter code for part 1 of the homework is available in the `1_cs231n` folder.

### Setup

Dependencies are listed in the `requirements.txt` file. If working with Anaconda, they should all be installed already.

Download data.

```bash
cd 1_cs231n/cs231n/datasets
./get_datasets.sh
```

### Q10.1: Softmax Regression (4 points)

Work through `softmax.ipynb` and implement the Softmax classifier. Here is a brief summary and if you need a detailed tutorial to brush up your knowledge, [this](http://cs231n.github.io/linear-classify/) is a nice place.

Before we go into the details of a classifier, let us assume that our training dataset consists of \\(N\\) instances \\(x\_i \in \mathbb{R}^D \\) of dimensionality \\(D\\).
Corresponding to each of the training instances,
we have labels \\(y\_i \in \{1,2,\dotsc ,K \}\\), where \\(K\\) is the number of classes.
In this homework, we are using the CIFAR-10 database where \\(N=50,000\\), \\(K=10\\), \\(D= 32 \times 32 \times 3\\)
(image of size  \\(32 \times 32\\) with \\(3\\) channels - Red, Green, and Blue).

Classification is the task of assigning a label to the input from a fixed set of categories or classes. A classifier consists of two important components:

**Score function:** This maps every instance \\(x_i\\) to a vector \\(z\_i\\) of dimensionality \\(K\\). Each of these entries represent the class scores for that image:

\\[ z\_i = Wx\_i + b \\]

Here, W is a matrix of weights of dimensionality \\(K \times D\\) and b is a vector of bias terms of dimensionality \\(K \times 1\\). The process of training is to find the appropriate values for W and b such that the score corresponding to the correct class is high. In order to do this, we need a function that evaluates the performance. Using this evaluation as feedback, the weights can be updated in the right 'direction' to improve the performance of the classifier.

Before proceeding, we'll incorporate the bias term into \\(W\\), making it of dimensionality \\(K \times (D+1)\\). Also let a superscript \\(j\\) denote the \\(j^{th}\\) element of \\(z\_i\\) and \\(w\_j\\) be the \\(j^{th}\\) row of W so that \\(z\_i^j = w\_j^Tx\_i\\). Finally apply the softmax function to compute probabilities (for the \\(i\\)th example and \\(j\\)th class):

\\[ p_i^j = \frac{e^{z\_i^{j}}}{\sum\_k e^{z^k\_i}} \\]

**Loss function:** This function quantifies the correspondence between the predicted scores and ground truth labels. Softmax regression uses the cross-entropy loss:

\\[ L = - \frac{1}{N}\sum\_{i=1}^{N}\log \left( p_i^{y_i} \right) \\]

If the weights are allowed to take values as high as possible, the model can overfit to the training data. To prevent this from happening a regularization term \\(R(W)\\) is added to the loss function. The regularization term is the squared some of the weight matrix \\(W\\). Mathematically,

\\[ R(W) = \sum\_{k}\sum\_{l}W\_{k,l}^2 \\]

The final loss is

\\[ \mathcal{L}(W) = L(W) + R(W) \\]

The regularization term \\(R(W)\\) is usually multiplied by the regularization strength \\(\lambda\\) before adding it to the loss function. \\(\lambda\\) is a hyper parameter which needs to be tuned so that the classifier generalizes well over the training set.

The next step is to update the weight parts such that the loss is minimized. This is done by Stochastic Gradient Descent (SGD). The weight update is done as:

\\[ W := W - \eta \nabla_W \mathcal{L}(W) \\]

Here, \\(\nabla_W \mathcal{L}\\) is the gradient of the loss function and the factor \\(\eta\\) is the learning rate. SGD is usually performed by computing the gradient w.r.t. a randomly selected batch from the training set.
This method is more efficient than computing the gradient w.r.t the whole training set before each update is performed.

### Q10.2: Two-layer Neural Network (4 points)

The IPython notebook `two_layer_net.ipynb` will walk you through implementing a
two-layer neural network on CIFAR-10. You will write a hard-coded 2-layer
neural network, implement its backward pass, and tune its hyperparameters.

### Q10.3: Modular Neural Network (6 points)

The IPython notebook `layers.ipynb` will walk you through a modular neural network implementation. You will implement the forward and backward passes of many different layer types.

## Part 2

This part is similar to the first part except that you will now be using [PyTorch][3] to 
implement the two-layer neural network. In part 1 you implemented core operations given significant scaffolding code. In part 2 these core operations are given by PyTorch and you simply need to figure out how to use them.

If you haven't already, install PyTorch (__please use PyTorch vesion >=0.2__). This will probably be as simple as running the
commands in the [Get Started][3] section of the PyTorch page, but if you run in to problems
check out the [installation section][10] of the github README, search Google, or come to
office hours. You may want to go through the [PyTorch Tutorial][12] before continuing.
This homework is not meant to provide a complete overview of Deep Learning framework
features or PyTorch features.

Open-source frameworks are becoming more and more
optimized and provide even faster implementations. Most of them take advantage of
both GPUs, which can offer a significant speedup (e.g., 50x). A library of highly optimized Deep
Learning operations from Nvidia called the [CUDA® Deep Neural Network library (cuDNN)][9]
also helps.

You will be using existing layers and hence, this part should be short and simple. To get
started with PyTorch you could just jump in to the implementation below or read through
some of the documentation below.

- What is PyTorch and what distinguishes it from other DL libraries? (github [README][11])
- PyTorch [Variables](http://pytorch.org/docs/master/autograd.html#variable) (needed for autodiff)
- PyTorch [Modules](http://pytorch.org/docs/master/nn.html)
- PyTorch [examples][8]

The necessary files for this section are provided in the `2_pytorch` directory.
You will only need to write code in `train.py` and in each file in the `models/` directory.

### Q10.4: Softmax Classifier using PyTorch (3 points)

The`softmax-classifier.ipynb` notebook will walk you through implementing a softmax
classifier using PyTorch. Data loading and scaffolding for a train loop are provided.
In `filter-viz.ipynb` you will load the trained model and extract its weight so they can be visualized.

### Q10.5: Two-layer Neural Network using PyTorch (3 points)

By now, you have an idea of working with PyTorch and may proceed to implementing a two-layer neural network. Go to 
`models/twolayernn.py` and complete the `TwoLayerNN` `Module`. Now train the neural network using

```bash
run_twolayernn.sh
```
    
You will need to adjust hyperparameters in `run_twolayernn.sh` to achieve good performance.
Use the code from `softmax-classifier.ipynb` to generate a __loss vs iterations__ plot for train
and val and a __validation accuracy vs iterations__ plot. Save these plots as `twolayernn_lossvstrain.png` and `twolayernn_valaccuracy.png` respectively

Make suitable modifications in `filter-viz.ipynb`
and save visualizations of the weights of the first hidden layer called `twolayernn_gridfilt.png`.

## Deliverables

Submit the deliverables for Q10 by uploading a zip file called `hw1_code.zip` to the `HW1 Code` section on Gradescope.

This zip could be generated by running the following script from the unzipped folder :

```bash
./collect_submission.sh
```
The following files should be included:

1. All the files originally in the 1_cs231n folder and the ones generated there (Part 1).
2. All the files originally in the 2_pytorch folder and the ones generated there (Part 2).
3. Model implementations `models/*.py` (Part 2).
4. Training code `train.py` (Part 2).
5. The shell scripts used to train the 2 models (`run_softmax.sh`, `run_twolayernn.sh`) (Part 2).
6. Learning curves (loss) and validation accuracy plots (Part 2).
7. The version of `filter-viz.ipynb` used to show filter visualizations (Part 2).
8. Log files for each model with test accuracy reported at the bottom (Part 2).

Note that the PDF being uploaded to the `HW1` section must also contain the Jupyter notebooks and marked according to the right sub-questions on Gradescope. You could append the PDFs generated from the Jupyter notebooks to the one containing your theory solutions and upload the resulting PDF on Gradescope.


References:

1. [CS231n Convolutional Neural Networks for Visual Recognition][2]

[2]: http://cs231n.stanford.edu/
[3]: http://pytorch.org/
[4]: http://bvlc.eecs.berkeley.edu/
[5]: https://www.cc.gatech.edu/classes/AY2021/cs7643_fall/
[8]: https://github.com/pytorch/examples
[9]: https://developer.nvidia.com/cudnn
[10]: https://github.com/pytorch/pytorch#installation
[11]: https://github.com/pytorch/pytorch
[12]: http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html


---

&#169; 2020 Georgia Tech
