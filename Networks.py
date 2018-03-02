import numpy as np
import random
import time

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

    def SGD(self, trainingDatas, testDatas, epochs, batchSize, eta, lambdaValue):

        start = time.clock()

        totalTrainDatasNumber = len(trainingDatas)
        #开始迭代
        for count in range(0, epochs):
            random.shuffle(trainingDatas)
            mini_trainDatas = [trainingDatas[k:k+batchSize]
                               for k in range(0, totalTrainDatasNumber, batchSize)]

            for mini_batch in mini_trainDatas:
                self.updata_para(mini_batch, eta, lambdaValue, totalTrainDatasNumber)

            #完成了一次迭代
            print("Epoch %s complete" % str(count) + "---Use Time : %s s" + str(time.clock() - start))
            start = time.clock()

            #使用测试数据计算正确率
            rightNumber = 0
            totalNumber = float(len(testDatas))
            for input, ideal in testDatas:

                out = self.moveforward(input)
                if(np.argmax(ideal) == np.argmax(out)):
                    rightNumber = rightNumber + 1

            print("Accuracy %s" % str(rightNumber/totalNumber))


    #更新参数值
    def updata_para(self, mini_trainDatas, eta, lambdaValue, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_trainDatas:
            delta_nable_b, delta_nable_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nable_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nable_w)]

        self.weights = [(1-eta*(lambdaValue/n))*w-(eta/len(mini_trainDatas))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_trainDatas))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    #反向传递函数
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        active = x
        actives = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, active) + b
            zs.append(z)
            #计算激活值
            active = sigmoid(z)
            #保存激活值
            actives.append(active)

        #交叉熵计算误差
        delta = CrossEntropyCost.delta(actives[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, actives[-2].transpose())

        #反向传递计算导数
        for l in range(2, self.num_layer):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.array(actives[-l - 1]).transpose())
        return (nabla_b, nabla_w)

class CrossEntropyCost:

     @staticmethod
     def cost(a, y):
         return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

     @staticmethod
     def delta(a, y):
         return (a-y)


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