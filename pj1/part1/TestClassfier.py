import math
import random
import matplotlib.pyplot as plt
import numpy as np
from ClassifierModel import ClassifierNet
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict


parser = argparse.ArgumentParser(description='Classifier Task')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
task_kind = "Classifier"
config = config[task_kind]

anno_test_path = config["Test"]["test_path"]
class_num = config["General"]["class_num"]
batch_size = config["Train"]["batch_size"]
# epochs = config["Train"]["epochs"]
net_arch = config["Train"]["net_arch"]
lr = config["Train"]["lr"]
random_range = config["Train"]["init_generation_random_range"]
is_load = config["Train"]["is_load"]
load_path = config["Test"]["load_path"]


def test(model):
    eval_dataset = MyDataset(annotation_path = anno_test_path,
                              class_num = class_num, )
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False)
    acc_num = 0
    total_loss = 0
    for i, batch in enumerate(eval_loader):
        img_tensor, label_tensor = batch # torch.tensor
        for j in range(0, min(len(label_tensor), batch_size)):
            img, label = img_tensor[j], label_tensor[j]
            img = img.numpy()
            pred = model.forward(img)
            pred_label = pred.argmax()+1
            if pred_label==label:
                acc_num+=1
            total_loss -= np.log(pred[label-1][0])

    acc_rate = acc_num / len(eval_dataset)
    avg_loss = total_loss / len(eval_dataset)
    print("eval_accuracy, %.2f total loss in %d data size" 
          % (total_loss, len(eval_dataset)))
    print("Avg_loss is %.2f, with acc_rate %.2f%% \n" % (avg_loss, (acc_rate*100)))


if __name__ == "__main__":
    
    net = ClassifierNet(net_arch = net_arch, 
                                 lr = lr, 
                                 random_range = random_range,
                                 batch_size = batch_size,
                                 task_kind = task_kind,
                                 is_load = is_load,
                                 load_path = load_path)
    net.load_network()
    test(net)
