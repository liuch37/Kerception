# Kerception Neural Networks
This is the prototype for the proposed Kerception neural network, tested with different DNN models/datasets using Tensorflow 2.0. Our implementation is on top of and modify the below repositories: 

https://github.com/amalF/Kervolution

https://github.com/CyberZHG/tf-keras-kervolution-2d

## Introduction
Our idea is based on the CVPR 2019 paper "Kervolutional Neural Networks" https://arxiv.org/pdf/1904.03955.pdf. The authors proposed a general operation for 2D convolution - kervolution. It is done by replacing linear weights in convolution with kernel tricks. Kernel representation can represent higher order statistics compared to linear weights, and hence it can learn more complicated features faster, especially in the early stage of a deep neural network.

## Idea
The paper discussed several classical kernels. However, it does not generalize how to set the hyperparameters for each kernel. It is almost impossible to tune every hyperparameter in one kernel, and needless to say it is impossible to select proper kernel manually when we would like to design a deep neural network architecture.

We propose the idea of "kerception" by combining "kervolution" and "inception". 

### Kerception Block
We combine the idea of kervolution with inception networks. We stack various kernels together and concatenate them into one kerception block - similar to the idea that inception network combines various convolutional kernel sizes and pooling layer.

#### Fully Trainable Polynomial Kernel
This does not work well - exponent is unlikely to be trained well. 

### Dynamic Scheduling for Constructing a Kerception Block
How to combine different kerception blocks still need more studies and ideas. To be dicovered.
