import pickle
import matplotlib.pyplot as plt
import numpy as np

#====================MNIST Performance=====================#
filename0 = './performance/MNIST_convolution_statistics.txt'
with open(filename0, 'rb') as fp:
    test_acc0 = pickle.load(fp)

filename1 = './performance/MNIST_kervolution_poly_statistics.txt'
with open(filename1, 'rb') as fp:
    test_acc1 = pickle.load(fp)

filename2 = './performance/MNIST_kervolution_sigmoid_statistics.txt'
with open(filename2, 'rb') as fp:
    test_acc2 = pickle.load(fp)

filename3 = './performance/MNIST_kervolution_gaussian_statistics.txt'
with open(filename3, 'rb') as fp:
    test_acc3 = pickle.load(fp)

filename4 = './performance/MNIST_kerceptionA_statistics.txt'
with open(filename4, 'rb') as fp:
    test_acc4 = pickle.load(fp)

filename5 = './performance/MNIST_kerv_conv_poly_trainable_statistics.txt'
with open(filename5, 'rb') as fp:
    test_acc5 = pickle.load(fp)

filename6 = './performance/MNIST_kervolution_poly_trainable_statistics.txt'
with open(filename6, 'rb') as fp:
    test_acc6 = pickle.load(fp)

filename7 = './performance/MNIST_kervolution_poly_hyper_statistics.txt'
with open(filename7, 'rb') as fp:
    test_acc7 = pickle.load(fp)

filename8 = './performance/MNIST_kerceptionB_statistics.txt'
with open(filename8, 'rb') as fp:
    test_acc8 = pickle.load(fp)


plt.figure(1)
plt.plot(np.arange(len(test_acc0)),test_acc0,'-',\
         np.arange(len(test_acc1)),test_acc1,'--',\
         np.arange(len(test_acc2)),test_acc2,'--',\
         np.arange(len(test_acc3)),test_acc3,'--',\
         np.arange(len(test_acc4)),test_acc4,'-.',\
         np.arange(len(test_acc5)),test_acc5,':',\
         np.arange(len(test_acc6)),test_acc6,'--',\
         np.arange(len(test_acc7)),test_acc7,'--',\
         np.arange(len(test_acc8)),test_acc8,'-.',\
         linewidth=2, markersize=8)
plt.ylabel('Validation Accuracy')
plt.xlabel('Training Step')
plt.grid()
plt.legend(['conv-conv','kerv-kerv: polynomial (dp=3, cp=1)','kerv-kerv: sigmoid','kerv-kerv: gaussian (g=1)','kerception A-conv (cp trainable)','kerv-conv: polynomial (dp=3, cp trainable)', 'kerv-kerv: polynomial (dp=3, cp trainable)','kerv-kerv: polynomial (dp=2, cp=1)','kerception B-conv (dp, cp trainable)'])
plt.title('MNIST')

#====================CIFAR10 Performance=====================#

filename9 = './performance/CIFAR10_resnet101_statistics.txt'
with open(filename9, 'rb') as fp:
    test_acc9 = pickle.load(fp)


plt.figure(2)
plt.plot(np.arange(len(test_acc9)),test_acc9,'-',\
         linewidth=2, markersize=8)
plt.ylabel('Validation Accuracy')
plt.xlabel('Training Step')
plt.grid()
plt.legend(['resnet-101'])
plt.title('CIFAR10')
plt.show()
