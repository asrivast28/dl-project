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
        # Following VGG16 architecture
        C, H, W = im_size;
        
        # Layer1
        V1 = 64;
        self.c1 = torch.nn.Conv2d(C, V1, kernel_size=3, padding=1);
        self.b1 = torch.nn.BatchNorm2d(V1);
        self.r1 = torch.nn.ReLU(inplace=True);

        # Layer2
        V1 = 64; V2 = 64;
        self.c2 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
        self.b2 = torch.nn.BatchNorm2d(V2);
        self.r2 = torch.nn.ReLU(inplace=True);
        self.p2 = torch.nn.MaxPool2d(kernel_size=2, stride=2);

        # Layer3
        V1 = 64; V2 = 128;
        self.c3 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
        self.b3 = torch.nn.BatchNorm2d(V2);
        self.r3 = torch.nn.ReLU(inplace=True);

        # Layer4
        V1 = 128; V2 = 128;
        self.c4 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
        self.b4 = torch.nn.BatchNorm2d(V2);
        self.r4 = torch.nn.ReLU(inplace=True);
        self.p4 = torch.nn.MaxPool2d(kernel_size=2, stride=2);

        # Layer5
       # V1 = 128; V2 = 256;
       # self.c5 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b5 = torch.nn.BatchNorm2d(V2);
       # self.r5 = torch.nn.ReLU(inplace=True);

       # # Layer6
       # V1 = 256; V2 = 256;
       # self.c6 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b6 = torch.nn.BatchNorm2d(V2);
       # self.r6 = torch.nn.ReLU(inplace=True);

       # # Layer7
       # V1 = 256; V2 = 256;
       # self.c7 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b7 = torch.nn.BatchNorm2d(V2);
       # self.r7 = torch.nn.ReLU(inplace=True);
       # self.p7 = torch.nn.MaxPool2d(kernel_size=2, stride=2);

       # # Layer8
       # V1 = 256; V2 = 512;
       # self.c8 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b8 = torch.nn.BatchNorm2d(V2);
       # self.r8 = torch.nn.ReLU(inplace=True);

       # # Layer9
       # V1 = 512; V2 = 512;
       # self.c9 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b9 = torch.nn.BatchNorm2d(V2);
       # self.r9 = torch.nn.ReLU(inplace=True);

       # # Layer10
       # V1 = 512; V2 = 512;
       # self.c10 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b10 = torch.nn.BatchNorm2d(V2);
       # self.r10 = torch.nn.ReLU(inplace=True);
       # self.p10 = torch.nn.MaxPool2d(kernel_size=2, stride=2);

       # # Layer11
       # V1 = 512; V2 = 512;
       # self.c11 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b11 = torch.nn.BatchNorm2d(V2);
       # self.r11 = torch.nn.ReLU(inplace=True);

       # # Layer12
       # V1 = 512; V2 = 512;
       # self.c12 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b12 = torch.nn.BatchNorm2d(V2);
       # self.r12 = torch.nn.ReLU(inplace=True);

       # # Layer13
       # V1 = 512; V2 = 512;
       # self.c13 = torch.nn.Conv2d(V1, V2, kernel_size=3, padding=1);
       # self.b13 = torch.nn.BatchNorm2d(V2);
       # self.r13 = torch.nn.ReLU(inplace=True);
       # self.p13 = torch.nn.MaxPool2d(kernel_size=2, stride=2);

        # Classifier
       # self.fc = torch.nn.Linear(512 * 1 * 1, n_classes);
        self.fc = torch.nn.Linear(512 * 4 * 4, n_classes);
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
        N = images.shape[0];
        X1 = self.r1(self.b1(self.c1(images)));
        X2 = self.p2(self.r2(self.b2(self.c2(X1))));

        X3 = self.r3(self.b3(self.c3(X2)));
        X4 = self.p4(self.r4(self.b4(self.c4(X3))));

        #X5 = self.r5(self.b5(self.c5(X4)));
        #X6 = self.r6(self.b6(self.c6(X5)));
        #X7 = self.p7(self.r7(self.b7(self.c7(X6))));

        #X8 = self.r8(self.b8(self.c8(X7)));
        #X9 = self.r9(self.b9(self.c9(X8)));
        #X10 = self.p10(self.r10(self.b10(self.c10(X9))));

        #X11 = self.r11(self.b11(self.c11(X10)));
        #X12 = self.r12(self.b12(self.c12(X11)));
        #X13 = self.p13(self.r13(self.b13(self.c13(X12))));

        #Z = self.fc(X13.reshape(N, -1));
        Z = self.fc(X4.reshape(N, -1));
        scores = Z.reshape(N, -1);
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

