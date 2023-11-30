import os
import torch
from torch.utils.data import Dataset

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

class MyDataset(Dataset):
    def __init__(self, path, word_dict, tag_dict):
        self.path = path
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        wordslist, tagslist = GetData(path)
        self.wordslist = wordslist
        self.tagslist = tagslist
        self.data = []
        # make str into index
        for i, words in enumerate(self.wordslist):
            words_idx = [self.word_dict.get(word, self.word_dict['_UNKNOW']) for word in words]
            tags_idx = [self.tag_dict[tag] for tag in self.tagslist[i]]
            self.data.append([words_idx, tags_idx])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collect_fn(self, batch):
        words_batch = [words for words, tags in batch]
        tags_batch = [tags for words, tags in batch]
        leng = [len(words) for words in words_batch]
        max_len = max(leng)
        words_batch = [words + [self.word_dict['_PAD']] * (max_len - len(words)) for words in words_batch]
        tags_batch = [tags + [self.tag_dict['O']] * (max_len - len(tags)) for tags in tags_batch]

        # dtype here must be torch.long
        words_batch = torch.tensor(words_batch, dtype=torch.long)
        tags_batch = torch.tensor(tags_batch, dtype=torch.long)
        leng = torch.tensor(leng, dtype=torch.long)

        return words_batch, tags_batch, leng