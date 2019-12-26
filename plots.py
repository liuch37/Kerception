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

plt.figure(1)
plt.plot(np.arange(len(test_acc0)),test_acc0,'-',\
         np.arange(len(test_acc1)),test_acc1,'--',\
         np.arange(len(test_acc2)),test_acc2,'-.',\
         np.arange(len(test_acc3)),test_acc3,'.',\
         linewidth=2, markersize=8)
plt.ylabel('Validation Accuracy')
plt.xlabel('Training Step')
plt.grid()
plt.legend(['convolution','kervolution - polynomial','kervolution - sigmoid','kervolution - gaussian'])
plt.show()