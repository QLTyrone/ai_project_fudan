import math
import numpy as np
from Layer import Layer


class RegressionNet(object):
    def __init__(self, net_arch=[1,64,64,1], lr=0.01, random_range=0.15, 
                 train_data_size=8000, batch_size=20, task_kind="Regression"):
        assert len(net_arch) >= 2, " ** Error!! 2 layers are needed at least!\n"

        self.net_arch = net_arch
        self.lr = lr
        self.random_range = random_range
        self.train_data_size = train_data_size
        self.train_data = np.linspace(-math.pi, math.pi, train_data_size)
        self.batch_size = batch_size
        self.task_kind = task_kind

        self.layers = []
        for i in range(0, len(self.net_arch)-1):
            if i==len(self.net_arch)-2:
                self.layers.append(Layer(self.net_arch[i], self.net_arch[i+1], 
                                           self.random_range, True, self.task_kind))
            else :
                # print(self.net_arch)
                self.layers.append(Layer(self.net_arch[i], self.net_arch[i+1], 
                                           self.random_range, False, self.task_kind))
    
    def forward(self, raw_input):
        for layer in self.layers:
            raw_input = layer.forward(raw_input)
        return raw_input
    
    def backward(self, loss):
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def update_weight(self, lr):
        for layer in self.layers:
            layer.update_weight(lr)

    def save_network(self):
        for i in range(len(self.net_arch) - 1):
            np.savetxt("./model/regress/weight_regress{}.txt".format(i), self.layers[i].get_weight())
            np.savetxt("./model/regress/bias_regress{}.txt".format(i), self.layers[i].get_bias())

    def load_network(self):
        for i in range(len(self.net_arch) - 1):
            # print(np.shape(np.loadtxt("./model/regress/bias_regress{}.txt".format(i)).reshape(self.layers[i].bias.shape[0], -1)))
            self.layers[i].set_weight(np.loadtxt("./model/regress/weight_regress{}.txt".format(i)).reshape(self.layers[i].weight.shape[0], -1))
            self.layers[i].set_bias(np.loadtxt("./model/regress/bias_regress{}.txt".format(i)).reshape(self.layers[i].bias.shape[0], -1))