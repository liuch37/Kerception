import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = './performance/MNIST_kervolution_poly_statistics.txt'
with open(filename, 'rb') as fp:
    test_acc = pickle.load(fp)

plt.figure(1)
plt.plot(np.arange(len(test_acc)),test_acc, '-', linewidth=2, markersize=8)
plt.ylabel('Validation Accuracy')
plt.xlabel('Training Step')
plt.grid()
plt.legend(['Kervolution - polynomial'])
plt.show()