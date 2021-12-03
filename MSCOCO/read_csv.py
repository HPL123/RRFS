import numpy as np
import os
import csv

images = []
path="/raid1/aditya/zero_shot_detection/zsd/MSCOCO/train_coco_seen_all.csv"
import csv
with open(path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        images.append(row[0])
        # print(row)
images = np.array(images)
import pdb;pdb.set_trace()
