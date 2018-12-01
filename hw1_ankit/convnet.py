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
        C, H, W = im_size;
        self.conv1 = torch.nn.Conv2d(C, hidden_dim, kernel_size);

        Hout = np.int64(1 + H - kernel_size);
        Wout = np.int64(1 + W - kernel_size);

        self.pool1 = torch.nn.MaxPool2d(2, stride=2);

        Hpool = np.int64(1 + ((Hout - 2) / 2));
        Wpool = np.int64(1 + ((Wout - 2) / 2));
        
        self.fc1 = torch.nn.Linear((hidden_dim * Hpool * Wpool), n_classes);
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
        N = images.shape[0];
        convrelu = self.conv1(images).clamp(min=0);
        poolout = self.pool1(convrelu);
        Z = self.fc1(poolout.reshape(N,-1));
        scores = Z;
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

