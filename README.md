# Kerception Neural Networks
This is the prototype for the proposed Kerception neural network and its dynamic scheduling algorithms, tested with different DNN models/datasets using Tensorflow 2.0. Our implementation is on top of https://github.com/amalF/Kervolution.

## Introduction
Our idea is based on the CVPR 2019 paper "Kervolutional Neural Networks" https://arxiv.org/pdf/1904.03955.pdf. The authors proposed a general operation for 2D convolution - kervolution. It is done by replacing linear weights in convolution with kernel tricks. Kernel representation can represent higher order statistics compared to linear weights, and hence it can learn more complicated features faster, especially in the early stage of a deep neural network.

## Idea
The paper discussed several classical kernels. However, it does not generalize how to set the hyperparameters for each kernel. It is almost impossible to tune every hyperparameter in one kernel, and needless to say it is impossible to select proper kernel manually when we would like to design a deep neural network architecture.

We propose several techniques to overcome the above drawback. 

### Kerception Block
We combine the idea of kervolution with inception networks. We stack various kernels together and concatenate them into one kerception block - similar to the idea that inception network combines various convolutional kernel sizes and pooling layer.

### Fully Trainable Polynomial Kernel
This does not work well - exponent is unlikely to be trained. 

### Dynamic Scheduling for Constructing a Kerception Block

### More to come......

## Target
We will target the below conference schedule depending on research progress.
1) ECCV 2020 (Deadline 2020/03/05)
2) BMVC 2020 (Deadline 2020/04/29 temp)
2) NIPS 2020 (Deadline 2020/05/23 temp)
3) ACCV 2020 (Deadline 2020/07/01 temp)
3) ICLR 2021 (Deadline 2020/09/25 temp)
4) CVPR 2021 (Deadline 2020/11/01 temp)
