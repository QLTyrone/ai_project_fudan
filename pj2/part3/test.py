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
embedding_size = config["Train"][kind]["embedding_size"]
hidden_dim = config["General"]["hidden_dim"]
load_path = config["Test"][kind]["load_path"]
batch_size = config["General"]["batch_size"]
out_path = config["Val"][kind]["out_path"]

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

def GetData(path):
    word_sentence_in_dataset = [] 
    tag_sentence_in_dataset = []
    word_in_sentence = []
    tag_in_sentence = []
    with open(path, "r", encoding="utf-8") as f:
        annotations = f.readlines()
    for annotation in annotations:
        word_and_tag = annotation.strip(" ").strip("\n").split(" ")
        if len(word_and_tag)<=1:
            word_sentence_in_dataset.append(word_in_sentence)
            tag_sentence_in_dataset.append(tag_in_sentence)
            tag_in_sentence = []
            word_in_sentence = []
            continue
        word = word_and_tag[0]
        tag = word_and_tag[1]
        word_in_sentence.append(word)
        tag_in_sentence.append(tag)
    word_sentence_in_dataset.append(word_in_sentence)
    tag_sentence_in_dataset.append(tag_in_sentence)
    return word_sentence_in_dataset, tag_sentence_in_dataset

def test(model, word_dict, tag_dict, val_words, device):
    model.eval()
    model.state = 'eval'

    f = open(out_path, "w", encoding="utf-8")
    for i, words in enumerate(val_words):
        seq_len = torch.tensor(len(words), dtype=torch.long).unsqueeze(0)
        seq_len = seq_len.to(device)
        word_idx = [word_dict[word] for word in words]
        word_idx = torch.tensor(word_idx, dtype=torch.long).unsqueeze(0)
        word_idx = word_idx.to(device)
        tags = model(word_idx, seq_len, tag_dict)
        for j in range(len(words)):
            f.write(words[j] + " " + [k for k, v in tag_dict.items() if v == tags[0][j]][0] + "\n")
        if i!=len(val_words)-1:
            f.write("\n")
    f.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    word_dict = GetDict([train_path, val_path])
    val_words, val_tags = GetData(val_path)
    tag_dict = config["Train"][kind]["tag_dict"]
    model = BiLSTM_CRF(embedding_size, hidden_dim, word_dict, tag_dict).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 

    print(" -------< Loading parameters from {} >------- \n".format(load_path))
    params = torch.load(load_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(params, strict=True) 

    test(model, word_dict, tag_dict, val_words, device)