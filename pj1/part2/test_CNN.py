r""" This file is to train and eval the classify-model """
r"""博1bo  学2xue  笃3du  志4zhi,
    切5qie 问6wen  近7jin 思8si, 
    自9zi  由10you 无11wu 用12yong """

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from CNNModel import CNNModel
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict


parser = argparse.ArgumentParser(description='Classifier Task With CNN')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["CNN"]

test_path = config["Test"]["annotation_path"]
class_num = config["General"]["class_num"]
batch_size = config["Train"]["batch_size"]
load_path = config["Test"]["load_path"]


def test(model):
    model.eval()
    eval_dataset = MyDataset(data_path = test_path,
                              class_num = class_num, )
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False)
    acc_num = 0
    for i, batch in enumerate(eval_loader):
        img_tensor, label_tensor = batch
        with torch.no_grad():
            pred_tensor = model(img_tensor.to(device))
            for j in range(pred_tensor.size(0)):
                pred = torch.argmax(pred_tensor[j]).item()
                if pred==label_tensor[j]:
                    acc_num+=1

    acc_rate = acc_num / len(eval_dataset)
    print("Acc_rate %.2f%% \n" % (acc_rate*100))


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModel()
    model.to(device)
    
    params = torch.load(load_path, map_location=device)
    model.load_state_dict(params, strict=True) 

    test(model)

