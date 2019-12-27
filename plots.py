import pickle
import matplotlib.pyplot as plt
import numpy as np

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

plt.figure(1)
plt.plot(np.arange(len(test_acc0)),test_acc0,'-',\
         np.arange(len(test_acc1)),test_acc1,'--',\
         np.arange(len(test_acc2)),test_acc2,'--',\
         np.arange(len(test_acc3)),test_acc3,'--',\
         np.arange(len(test_acc4)),test_acc4,'-.',\
         np.arange(len(test_acc5)),test_acc5,':',\
         np.arange(len(test_acc6)),test_acc6,'--',\
         linewidth=2, markersize=8)
plt.ylabel('Validation Accuracy')
plt.xlabel('Training Step')
plt.grid()
plt.legend(['conv-conv','kerv-kerv: polynomial (fixed)','ker -kerv: sigmoid (fixed)','kerv-kerv: gaussian (fixed)','kerception A-conv (trainable)','kerv-conv: polynomial (trainable)', 'kerv-kerv: polynomial (trainable)'])
plt.title('MNIST')
plt.show()