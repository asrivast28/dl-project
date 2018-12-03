# NOTE: The scaffolding code for this part of the assignment
# is adapted from https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from cifar100 import CIFAR100

# You should implement these (softmax.py, twolayernn.py, convnet.py)
import mymodel

# Training settings
parser = argparse.ArgumentParser(description='CIFAR-10 Example')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['mymodel'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--cifar100-dir', default='data/100',
                    help='directory that contains cifar-100-python/ '
                         '(downloaded automatically if necessary)')
parser.add_argument('--permute-labels', action='store_true', default=False,
                    help='randomly permute class labels before every epoch')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load CIFAR100 using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# CIFAR100 meta data
n_classes = 100
im_size = (3, 32, 32)
# Subtract the mean color and divide by standard deviation. The mean image
# from part 1 of this homework was essentially a big gray blog, so
# subtracting the same color for all pixels doesn't make much difference.
# mean color of training images
cifar100_mean_color = [0.49131522, 0.48209435, 0.44646862]
# std dev of color across training images
cifar100_std_color = [0.01897398, 0.03039277, 0.03872553]
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(cifar100_mean_color, cifar100_std_color),
            ])
# Datasets
train_dataset = CIFAR100(args.cifar100_dir, split='train', download=False,
                        transform=transform)
val_dataset = CIFAR100(args.cifar100_dir, split='val', download=False,
                        transform=transform)
test_dataset = CIFAR100(args.cifar100_dir, split='test', download=False,
                        transform=transform)
# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)

# Load the model
model = mymodel.MyModel(im_size, args.hidden_dim,
                        args.kernel_size, n_classes)
if args.cuda:
    model.cuda()

#############################################################################
# TODO: Initialize an optimizer from the torch.optim package using the
# appropriate hyperparameters found in args. This only requires one line.
#############################################################################
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

# initial checkpoint
# model.checkpoint()

def update_centroids(centroids):
    '''
    Get true cluster centroids, using true labels as cluster labels.
    '''
    print('Updating centroids...')
    with torch.no_grad():
        all_labels = np.asarray(train_dataset.train_labels)
        for label in sorted(set(train_dataset.train_labels)):
            label_indices = np.where(all_labels == label)[0]
            label_loader = torch.utils.data.DataLoader(train_dataset,
                             batch_size=label_indices.size, sampler=torch.utils.data.sampler.SubsetRandomSampler(label_indices),
                             shuffle=False, **kwargs)
            label_images = next(iter(label_loader))[0]
            if args.cuda:
                label_images = label_images.cuda()
            output = F.softmax(model(label_images), dim=1)
            centroids[label] = torch.mean(output, dim=0)

# initial centroids 
centroids = torch.cuda.FloatTensor(n_classes, n_classes) if args.cuda else torch.FloatTensor(n_classes, n_classes)

def kl_from_centroids(outputs, targets, **kwargs):
    '''
    Use the KL-Divergence from centroids to calculate the loss.
    '''
    # print(centroids)
    log_probs = F.log_softmax(outputs, dim=1)
    return F.kl_div(log_probs, centroids[targets], **kwargs)


criterion = kl_from_centroids


def train(epoch, permutation):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    # train loop
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        images, targets = Variable(batch[0]), Variable(permutation[batch[1]])
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets, size_average=False) / len(images)
        loss.backward()
        optimizer.step()
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', permutation, n_batches=4)
            train_loss = loss.data[0]
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))
            # print(str(epoch) + ',' + ','.join(str(g) for g in model.compare_weights()))
            # print(str(epoch) + ',' + ','.join(str(g) for g in model.level_grads()))


def pairwise_kl(outputs, centroids):
    '''
    Calculate pairwise distance between all pairs of outputs and centroids, using KL-divergence.
    '''
    log_probs = F.log_softmax(outputs, dim=1)
    log_div = log_probs.view(outputs.shape[0], 1, outputs.shape[1]) - torch.log(centroids)
    kl_div = -1.0 * torch.sum(log_div * centroids.view(1, centroids.shape[0], centroids.shape[1]), dim=2)
    return kl_div


def evaluate(split, permutation, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        target = permutation[target]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        outputs = model(data)
        loss += criterion(outputs, target, size_average=False).data[0]
        # print('pairwise_kl', pairwise_kl(output, centroids)[10, 15])
        # print('kl_div', F.kl_div(F.log_softmax(output[10]), centroids[15], size_average=False))

        # predict the label as that of the nearest centroid
        pred = torch.argmin(pairwise_kl(outputs, centroids), dim=1)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


permutation = torch.arange(n_classes, dtype=torch.int64)
# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    if args.permute_labels:
        permutation = torch.randperm(n_classes)
    update_centroids(centroids)
    train(epoch, permutation)
evaluate('test', permutation, verbose=True)

# Save the model (architecture and weights)
torch.save(model, args.model + '.pt')
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details

