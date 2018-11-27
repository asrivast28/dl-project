#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 32 \
    --epochs 20 \
    --weight-decay 0.01 \
    --momentum 0.9 \
    --batch-size 200 \
    --seed 0 \
    --lr 0.000278 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
