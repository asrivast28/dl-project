import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        C, H, W = im_size
        weight_scale = 1e-4

        self.kernel_size = kernel_size
        self.padding_c = (kernel_size - 1) // 2
        self.stride_c = 1

        self.pool_size = 2
        self.padding_m = 0
        self.stride_m = 2

        self.N = 3
        self.M = 2

        self.W_c = nn.ParameterList([None for n in range(self.N)])
        self.b_c = nn.ParameterList([None for n in range(self.N)])
        for n in range(self.N):
            self.W_c[n] = nn.Parameter(torch.randn(hidden_dim, C, kernel_size, kernel_size) * weight_scale, requires_grad=True)
            self.b_c[n] = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            C = hidden_dim
        # self.W_c = nn.Parameter(torch.randn(hidden_dim, C, kernel_size, kernel_size) * weight_scale, requires_grad=True)
        # self.b_c = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        self.W_a = nn.ParameterList([None for m in range(self.M)])
        self.b_a = nn.ParameterList([None for m in range(self.M)])
        hidden_dims = [None for m in range(self.M)]
        hidden_dims[0] = hidden_dim * H * W // (4 ** self.N)
        for m in range(1, self.M):
            hidden_dims[m] = 10000 // 10

        for m in range(self.M-1):
            self.W_a[m] = nn.Parameter(torch.randn(hidden_dims[m], hidden_dims[m+1]) * weight_scale, requires_grad=True)
            self.b_a[m] = nn.Parameter(torch.zeros(hidden_dims[m+1]), requires_grad=True)
        self.W_a[self.M-1] = nn.Parameter(torch.randn(hidden_dims[-1], n_classes) * weight_scale, requires_grad=True)
        self.b_a[self.M-1] = nn.Parameter(torch.zeros(n_classes), requires_grad=True)
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
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        p = images
        for n in range(self.N):
            c = F.conv2d(p, self.W_c[n], bias=self.b_c[n], stride=self.stride_c, padding=self.padding_c)
            r = F.relu(c)
            p = F.max_pool2d(r, self.pool_size, stride=self.stride_m, padding=self.padding_m)
        # c = F.conv2d(p, self.W_c[-1], bias=self.b_c[-1], stride=self.stride_c, padding=self.padding_c)
        # r = F.relu(c)
        # p = r
        # p = F.max_pool2d(r, self.pool_size, stride=self.stride_m, padding=self.padding_m)

        previous = p.view(p.size(0), self.W_a[0].size(0))
        for m in range(self.M):
            scores = previous.matmul(self.W_a[m]) + self.b_a[m]
            previous = F.relu(scores)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
