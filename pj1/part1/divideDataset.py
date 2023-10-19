import os
import random

dataset_path = "../dataset"
train_path = "dataPath/train_data.txt"
test_path = "dataPath/test_data.txt"
val_path = "dataPath/val_data.txt"

class_num = 12
num_in_class = 620

train_file = open(train_path, "w")
test_file = open(test_path, "w")
val_file = open(val_path, "w")
for i in range(1, class_num+1):
    test_num = 0
    val_num = 0
    for j in range(1, num_in_class+1):
        save_file = os.path.join(dataset_path, str(i), str(j)+".bmp")
        randamx = random.randint(0, 100)
        if test_num < num_in_class//10 and randamx < 20:
            # test
            test_file.write(save_file+ " %d \n" % i)
            test_num += 1
        if val_num < num_in_class//10 and randamx >= 90:
            # val
            val_file.write(save_file+ " %d \n" % i)
            val_num += 1
        else:
            # train
            train_file.write(save_file+ " %d \n" % i)
