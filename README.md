# Deep Learning Project - Fall 2018

This repository is meant for collaborating on the Deep Learning project on [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520), the code for which is available on [GitHub](https://github.com/facebookresearch/deepcluster).

Following is a description of the directories/files in this repository (please update it whenever you add new material):

* **hw1_ankit** This directory contains a modified version of Ankit's [HW1](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/) submission. In particular, the *train.py* takes a new argument called `--permute-labels` which randomly permutes the label identifiers at the beginning of every epoch. This is used to test the robustness of the neural network against label permutations.
