#!/bin/bash
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
seed=$1
labels=$2
model=resnet
train=train_100.py

if [ "$model" = "mymodel" ]; then
python -u $train \
    --model mymodel \
    --kernel-size 7 \
    --hidden-dim 512 \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed 0 \
    --lr 0.1 | tee ${model}.log
fi

if [ "$model" = "resnet" ];
then

if [ "$labels" = "perm" ];
then
python -u $train \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 256 \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed ${seed} \
    --permute-labels \
    --lr 0.1 | tee ${model}_${seed}_${labels}.log
else
python -u $train \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 256 \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed ${seed} \
    --lr 0.1 | tee ${model}_${seed}_${labels}.log
fi

fi

if [ "$model" = "vgg" ];
then

if [ "$labels" = "perm" ];
then
python -u $train \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 256 \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed ${seed} \
    --permute-labels \
    --lr 0.1 | tee ${model}_${seed}_${labels}.log
else
python -u $train \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 256 \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --seed ${seed} \
    --lr 0.1 | tee ${model}_${seed}_${labels}.log
fi

fi
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
