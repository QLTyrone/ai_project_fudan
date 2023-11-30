import numpy as np
import argparse
import yaml
from easydict import EasyDict
from collections import OrderedDict

parser = argparse.ArgumentParser(description='HMM')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["HMM"]
kind = "Cn"
# kind = "En"
train_path = config["Train"][kind]["train_path"]
val_path = config["Val"][kind]["val_path"]
out_path = config["Val"][kind]["out_path"]

def GetDict(path_lists):
    word_dict = OrderedDict()
    for path in path_lists:
        with open(path, "r", encoding="utf-8") as f:
            annotations = f.readlines()
        for annotation in annotations:
            word_and_tag = annotation.strip(" ").split(" ")
            if len(word_and_tag)<=1:
                continue
            word = word_and_tag[0]
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

class HMMModel():
    def __init__(self, word_dict, tag_dict, train_words, train_tags):
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.words = train_words
        self.tags = train_tags
        
        self.trans = np.zeros((len(tag_dict), len(tag_dict)))
        self.emits = np.zeros((len(tag_dict), len(word_dict)))
        self.inits = np.zeros(len(tag_dict))

        for i, tags in enumerate(self.tags):
            for j, tag in enumerate(tags):
                w = self.words[i][j]
                self.emits[tag_dict[tag]][word_dict[w]] += 1
                if j==0:
                    self.inits[tag_dict[tag]] += 1
                if j==len(tags)-1:
                    pass
                else :
                    next_tag = tags[j+1]
                    self.trans[tag_dict[tag]][tag_dict[next_tag]] += 1

        self.inits = self.inits / self.inits.sum()
        self.inits[self.init==0] = 1e-8
        self.inits = np.log10(self.inits)

        for i in range(0, len(self.trans)):
            sum = self.trans[i].sum()
            if sum==0:
                self.trans[i] = 0
            else :
                self.trans[i] = self.trans[i] / sum
            self.trans[i][self.trans[i]==0] = 1e-8
            self.trans[i] = np.log10(self.trans[i])
        
        for i in range(0, len(self.emits)):
            sum = self.emits[i].sum()
            if sum==0:
                self.emits[i] = 0
            else :
                self.emits[i] = self.emits[i] / sum
            self.emits[i][self.emits[i]==0] = 1e-8
            self.emits[i] = np.log10(self.emits[i])

    def predict(self, val_words, out_path):
        f = open(out_path, "w", encoding="utf-8")
        for i, words in enumerate(val_words):
            prob = np.zeros((len(words), len(self.tag_dict)))
            max_prob_pos = np.zeros((len(words), len(self.tag_dict)))
            states = np.zeros(len(words))    
            prob[0] = self.inits + self.emits[:, self.word_dict[words[0]]]
            for j in range(1, len(words)):
                max_prob = prob[j-1] + self.trans.T
                max_prob_pos[j] = np.argmax(max_prob, axis=1)
                prob[j] = [max_prob[k, int(max_prob_pos[j][k])] for k in range(max_prob.shape[0])]
                prob[j] = prob[j] + self.emits[:, self.word_dict[words[j]]]
            
            states[-1] = np.argmax(prob[-1])
            for j in reversed(range(0, len(prob)-1)):
                states[j] = max_prob_pos[j+1][int(states[j+1])]
            rev_tag_dict = list(self.tag_dict.keys())
            for j in range(len(states)):
                f.write(words[j] + " " + rev_tag_dict[int(states[j])] + "\n")
            if i!=len(val_words)-1:
                f.write("\n")

        f.close()


if __name__ == "__main__":

    word_dict = GetDict([train_path, val_path])
    tag_dict = config["Train"][kind]["tag_dict"]

    train_words, train_tags = GetData(train_path)
    val_words, val_tags = GetData(val_path)

    HMM = HMMModel(word_dict, tag_dict, train_words, train_tags)
    HMM.predict(val_words, out_path)



    

    
    

