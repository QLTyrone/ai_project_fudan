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
config = config["Classifier"]

train_path = config["Train"]["train_path"]
val_path = config["Val"]["val_path"]
class_num = config["General"]["class_num"]
batch_size = config["Train"]["batch_size"]
epochs = config["Train"]["epochs"]
net_arch = config["Train"]["net_arch"]
lr = config["Train"]["lr"]
random_range = config["Train"]["init_generation_random_range"]
is_load = config["Train"]["is_load"]
load_path = config["Train"]["load_path"]


def eval(model):
    eval_dataset = MyDataset(annotation_path = val_path,
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
    return avg_loss, acc_rate


if __name__ == "__main__":
    
    net = ClassifierNet(net_arch = net_arch, 
                                 lr = lr, 
                                 random_range = random_range,
                                 batch_size = batch_size,
                                 is_load = is_load,
                                 load_path = load_path)
    
    train_dataset = MyDataset(annotation_path = train_path,
                              class_num = class_num, )
    
    train_loader = DataLoader(dataset = train_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = True)
    best_acc = 0
    epoch_record_x = []
    avg_loss_record_y = []
    acc_rate_record_y = []
    epoch_record_x.append(0)
    avg_loss_y, acc_rate_y = eval(net)
    avg_loss_record_y.append(avg_loss_y)
    acc_rate_record_y.append(acc_rate_y)

    for epoch in range(0, epochs+1):
        for i, batch in enumerate(train_loader):
            img_tensor, label_tensor = batch # torch.tensor
            for j in range(0, batch_size):
                img, label = img_tensor[j], label_tensor[j]
                img = img.numpy()
                pred = net.forward(img)
                gt_one_hot = [0]*class_num
                gt_one_hot[label-1] = 1
                gt_one_hot = np.array(gt_one_hot).reshape((class_num,1))
                loss = pred - gt_one_hot
                net.backward(loss)
            net.update_weight(net.lr)
        
        if epoch % 10 == 0:
            print("Epoch" , epoch)
            if_draw = False
            epoch_record_x.append(epoch)
            avg_loss_y, acc_rate_y = eval(net)
            
            if best_acc < acc_rate_y:
                net.save_network()
                best_acc = acc_rate_y

            avg_loss_record_y.append(avg_loss_y)
            acc_rate_record_y.append(acc_rate_y)
    
    plt.plot(epoch_record_x, avg_loss_record_y,
                color="red", label="avg_loss_record")
    plt.plot(epoch_record_x, acc_rate_record_y, 
                color="green", label="acc_rate_record")
    plt.title("Loss with epoch")
    plt.legend()
    plt.show()
