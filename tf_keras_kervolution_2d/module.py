import tensorflow as tf
from kernels import LinearKernel      # Equivalent to normal convolution
from kernels import L1Kernel          # Manhattan distance
from kernels import L2Kernel          # Euclidean distance
from kernels import PolynomialKernel  # Polynomial
from kernels import GaussianKernel    # Gaussin / RBF
from kernels import SigmoidKernel     # Sigmoid
from layers import KernelConv2D       # Basis Kervolution

__all__ = ['Kerception_blockC']

class Kerception_blockC(tf.keras.layers.Layer):
    '''
    Customized kervolution 2D + ratio proportional [0.1, 0.1, 0.2, 0.3, 0.3] inception block with total 16 filters.
    '''
    def __init__(self):

        super(Kerception_blockC,self).__init__()
        self.kernel_fn1 = LinearKernel()
        self.kconv1 = KernelConv2D(filters=1, kernel_size=3, padding='same', kernel_function=self.kernel_fn1)
        self.kernel_fn2 = SigmoidKernel()
        self.kconv2 = KernelConv2D(filters=1, kernel_size=3, padding='same', kernel_function=self.kernel_fn2)
        self.kernel_fn3 = GaussianKernel(gamma=1.0, trainable_gamma=True, initializer='he_normal')
        self.kconv3 = KernelConv2D(filters=4, kernel_size=3, padding='same', kernel_function=self.kernel_fn3)
        self.kernel_fn4 = PolynomialKernel(p=3, trainable_c=True)
        self.kconv4 = KernelConv2D(filters=5, kernel_size=3, padding='same', kernel_function=self.kernel_fn4)
        self.kernel_fn5 = PolynomialKernel(p=5, trainable_c=True)
        self.kconv5 = KernelConv2D(filters=5, kernel_size=3, padding='same', kernel_function=self.kernel_fn5)

    def call(self, x):
        x1 = self.kconv1(x)
        x2 = self.kconv2(x)
        x3 = self.kconv3(x)
        x4 = self.kconv4(x)
        x5 = self.kconv5(x)

        return tf.keras.layers.concatenate([x1, x2, x3, x4, x5], axis = 3)