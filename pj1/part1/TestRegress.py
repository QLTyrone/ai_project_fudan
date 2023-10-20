from RegressModel import RegressionNet
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import math
import random
from easydict import EasyDict


parser = argparse.ArgumentParser(description='Regression Task')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["Regression"]

train_data_size = config["Train"]["data_size"]
x_list_size = config["Val"]["data_size"]
batch_size = config["Train"]["batch_size"]
epochs = config["Train"]["epochs"]
net_arch = config["Train"]["net_arch"]
lr = config["Train"]["lr"]
random_range = config["Train"]["init_generation_random_range"]

def test(model, test_data_size=500):
    x_list = []
    for i in range(0, test_data_size):
        x_list.append(random.uniform(-math.pi, math.pi))
    x_list.sort()
    y_list = []
    pred_list = []
    total_loss = 0
    for i in range(0, test_data_size):
        x = x_list[i]
        y_list.append(np.sin(x))
        pred_list.append(model.forward(np.array([[x]])))
        total_loss += abs(y_list[i] - pred_list[i])
        # print(model.forward(np.array([[x]])))
    avg_loss = total_loss / test_data_size
    print("avg_loss =", avg_loss)
    plt.figure()
    plt.scatter(x_list, pred_list, s=5)
    plt.scatter(x_list, y_list, s=5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    net = RegressionNet(net_arch = net_arch, 
                          lr = lr, 
                          train_data_size = train_data_size,
                          random_range = random_range,
                          batch_size = batch_size)
    net.load_network()
    test(net)