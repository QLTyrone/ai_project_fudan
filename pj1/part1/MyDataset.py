import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, annotation_path, class_num, ):
        super(MyDataset, self).__init__()
        self.img_paths = []
        self.img_labels = []
        with open(annotation_path, "r") as f:
            paths = f.readlines()
        self.sample_num = len(paths)

        for annotation in paths:
            splited_string = annotation.strip(" ").split(" ")
            img_path = splited_string[0]
            img_label = int(splited_string[1])
            assert img_label <= class_num, "Error in annotation file!!"
            self.img_paths.append(img_path)
            self.img_labels.append(img_label)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('L')
        t = transforms.ToTensor()
        # flatten img into one dim
        img = t(img).numpy().reshape((28*28,1)) 
        label = self.img_labels[index]
        return img, label
    
    def __len__(self):
        return self.sample_num
    
    