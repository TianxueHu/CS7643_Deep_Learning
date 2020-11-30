import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.channels, self.height, self.width = im_size
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(in_channels = self.channels, out_channels = self.hidden_dim, kernel_size = self.kernel_size) 
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = self.kernel_size, stride = 1) ###
        self.conv_out_size = int(self.hidden_dim * (self.height - 2*kernel_size+2) * (self.width - 2*kernel_size+2))
        self.layer1 = nn.Linear(self.conv_out_size, n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #N, C, W, H = images.shape
        conv_out = self.conv1(images)
        conv_out = self.relu1(conv_out)
        conv_out = self.maxpool1(conv_out)
        fc_1 = conv_out.view(-1, self.conv_out_size)
        scores = self.layer1(fc_1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

