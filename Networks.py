import numpy as np


class Network:

    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.weight_init()

    #初始化权重和偏移值
    def weight_init(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    #前向传递计算
    def moveforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a


"""
激活函数
"""
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

"""
激活函数导数函数
"""
def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))