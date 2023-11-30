import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from itertools import chain
import argparse
import yaml
from easydict import EasyDict
from collections import OrderedDict
import matplotlib.pyplot as plt

from MyDataset import MyDataset
from BiLSTM_CRF import BiLSTM_CRF

parser = argparse.ArgumentParser(description='BiLSTM+CRF')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["BiLSTM_CRF"]
kind = "Cn" #### 
# kind = "En" #### 
train_path = config["Train"][kind]["train_path"]
val_path = config["Val"][kind]["val_path"]
out_path = config["Val"][kind]["out_path"]
batch_size = config["General"]["batch_size"]
epochs = config["General"]["epochs"]
embedding_size = config["Train"][kind]["embedding_size"]
hidden_dim = config["General"]["hidden_dim"]
is_load = config["General"]["is_load"]
load_path = config["General"]["load_path"]
save_path = config["General"]["save_path"]


def GetDict(path_lists):
    word_dict = OrderedDict()
    word_dict["_PAD"] = 0
    word_dict["_UNKNOW"] = 1
    for path in path_lists:
        with open(path, "r", encoding="utf-8") as f:
            annotations = f.readlines()
        for annotation in annotations:
            splited_string = annotation.strip(" ").split(" ")
            if len(splited_string)<=1:
                continue
            word = splited_string[0]
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict


def eval(model, word_dict, tag_dict):
    model.eval()
    model.state = 'eval'
    all_label = []
    all_pred = []
    # word_dict_rev = {k:v for v,k in word_dict.items()}
    tag_dict_rev = {k:v for v,k in tag_dict.items()}
    print(" -------< Evaluating >------- \n")
    eval_dataset = MyDataset(path = val_path,
                              word_dict = word_dict,
                              tag_dict = tag_dict, )
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False,
                              collate_fn = eval_dataset.collect_fn)
    for i, batch in enumerate(eval_loader):
        words_batch, tags_batch, leng = batch # torch.tensor
        words_batch = words_batch.to(device)
        tags_batch = tags_batch.to(device)
        leng = leng.to(device)
        with torch.no_grad():
            batch_tag = model(words_batch, leng, tags_batch)
            all_label.extend([[tag_dict_rev[t] for t in l[:leng[i]].tolist()] for i, l in enumerate(tags_batch)])
            all_pred.extend([[tag_dict_rev[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.tag_dict.keys()]
    f1 = metrics.f1_score(all_label, all_pred, average='micro', labels=sort_labels[1:-3], zero_division=1)
    # print(metrics.classification_report(all_label, all_pred, labels=sort_labels[:-3], digits=3))
    print("micro_avg f1-score: %.4f" % f1)
    model.train()
    model.state = 'train'
    return f1


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    word_dict = GetDict([train_path, val_path])
    tag_dict = config["Train"][kind]["tag_dict"]
    model = BiLSTM_CRF(embedding_size, hidden_dim, word_dict, tag_dict).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 

    if is_load:
        print(" -------< Loading parameters from {} >------- \n".format(load_path))
        params = torch.load(load_path, map_location='cuda:0')
        model.load_state_dict(params, strict=True) 

    train_dataset = MyDataset(path = train_path,
                              word_dict = word_dict,
                              tag_dict = tag_dict, )
    train_loader = DataLoader(dataset = train_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False,
                              collate_fn = train_dataset.collect_fn)
    
    best_f1 = 0.
    epoch_record_x = []
    micro_avg_score_y = []
    epoch_record_x.append(0)
    score = eval(model, word_dict, tag_dict)
    micro_avg_score_y.append(score)

    for epoch in range(1, epochs+1):
        avg_loss = 0.
        for i, batch in enumerate(train_loader):
            words_batch, tags_batch, leng = batch # torch.tensor
            words_batch = words_batch.to(device)
            tags_batch = tags_batch.to(device)
            leng = leng.to(device)
            loss = model(words_batch, leng, tags_batch)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch" , epoch)
        print("Train Loss: ", avg_loss/len(train_dataset))

        if epoch % 5 == 0:
            
            epoch_record_x.append(epoch)
            score = eval(model, word_dict, tag_dict)
            micro_avg_score_y.append(score)
            
            if best_f1 < score:
                print(" -------< Saved Best Model >------- \n")
                torch.save(model.state_dict(), save_path)
                best_f1 = score

            if epoch in [30,50]:
                lr = optimizer.param_groups[0]['lr']
                print(" -------< Reducing lr to {} >------- \n".format(lr/2))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr/2
        
    
    plt.plot(epoch_record_x, micro_avg_score_y, 
                color="green", label="mirco_avg_f1_record")
    plt.title("f1 with epoch")
    plt.legend()
    plt.show()

