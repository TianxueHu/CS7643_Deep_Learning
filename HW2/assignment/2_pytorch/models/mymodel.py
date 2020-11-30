import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        '''
        self.conv1 = nn.Conv2d(im_size[0], hidden_dim, kernel_size, padding = (kernel_size-1)//2) 
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding = (kernel_size-1)//2)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) 
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding = (kernel_size - 1)//2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.layer1 = nn.Linear(hidden_dim * 256, hidden_dim * 32)
        self.layer2 = nn.Linear(hidden_dim * 32, hidden_dim * 8)
        self.layer3 = nn.Linear(hidden_dim * 8, n_classes)
        '''
        self.pretrained = models.resnet18(pretrained=True)
        n_features = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(in_features = n_features, out_features = 10, bias = True)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################
        '''
        scores = self.conv1(images)
        scores = self.relu1(scores)
        scores = self.conv2(scores)
        scores = self.relu2(scores)
        scores = self.pool(scores)
        scores = self.conv3(scores)
        scores = self.relu3(scores)

        scores = self.layer1(scores.reshape((images.shape[0], -1)))
        relu_l1 = nn.ReLU()
        scores = relu_l1(scores)
        scores = self.layer2(scores)
        relu_l2 = nn.ReLU()
        scores = relu_l2(scores)
        scores = self.layer3(scores)
        '''
        images = F.upsample(images, size=(128, 128), mode='bilinear')
        scores = self.pretrained(images)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

