import DataDispose
import Networks
import numpy as np

#学习速率
studySpeed = 0.002
#规范化参数
lambdaValue = 10
#隐藏神经元数目
hideNeurnNumber = 10000
#一次处理的数目
batchSize = 10
#迭代次数
epochs = 400

train, test = DataDispose.read_csv_data()
net = Networks.Network([7, hideNeurnNumber, 2])
net.SGD(train, test, epochs, batchSize, studySpeed, lambdaValue)

np.save("Para/biases.npy", net.biases)
np.save("Para/weights.npy", net.weights)

print("Saved the para")

