import numpy as np
import argparse
import yaml
from easydict import EasyDict
from collections import OrderedDict
from sklearn_crfsuite import CRF


parser = argparse.ArgumentParser(description='CRF')
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args()
config_path =args.config_path
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
config = config["CRF"]
# kind = "Cn"
kind = "En"
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

def Word2Features(sent, i):
    word = sent[i]
    pre_word = '<s>' if i == 0 else sent[i-1]
    nxt_word = '</s>' if i == (len(sent)-1) else sent[i+1]
    features = {
        'w': word,
        'w-1': pre_word,
        'w+1': nxt_word,
        'w-1:w': pre_word + word,
        'w:w+1': word + nxt_word,
        'bias': 1
    }
    return features

def Sentence2Features(sentence):
    return [Word2Features(sentence, i) for i in range(len(sentence))]

class CRFModel(object):
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, 
                 max_iterations=100, all_possible_transitions=False):
        self.crf = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, train_words, train_tags):
        feature_by_sentence = [Sentence2Features(s) for s in train_words]
        self.crf.fit(feature_by_sentence, train_tags)

    def predict(self, val_words, out_path):
        f = open(out_path, "w", encoding="utf-8")
        feature_by_sentence = [Sentence2Features(s) for s in val_words]
        preds = self.crf.predict(feature_by_sentence)
        for i, words in enumerate(val_words):
            for j in range(len(words)):
                f.write(words[j] + " " + preds[i][j] + "\n")
            if i!=len(val_words)-1:
                f.write("\n")
        f.close()


if __name__ == "__main__":

    word_dict = GetDict([train_path, val_path])
    tag_dict = config["Train"][kind]["tag_dict"]

    train_words, train_tags = GetData(train_path)
    val_words, val_tags = GetData(val_path)

    crf = CRFModel()
    crf.train(train_words, train_tags)
    crf.predict(val_words, out_path)
