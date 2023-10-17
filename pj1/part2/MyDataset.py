import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_path, class_num, ):
        super(MyDataset, self).__init__()
        self.img_paths = []
        self.img_labels = []
        with open(data_path, "r") as f:
            paths = f.readlines()
        self.sample_num = len(paths)
        self.class_num = class_num

        for path in paths:
            splited_string = path.strip(" ").split(" ")
            img_path = splited_string[0]
            img_label = int(splited_string[1])
            assert img_label <= class_num, "Error in annotation file!!"
            self.img_paths.append(img_path)
            self.img_labels.append(img_label)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('L')
        trans = transforms.ToTensor()
        img = trans(img)
        label_num = self.img_labels[index]-1
        return img, label_num
    
    def __len__(self):
        return self.sample_num
    
    