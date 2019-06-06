import os
from tqdm import tqdm
import numpy as np 

dir_path = '../../datasets/humanparsing/'
val_size = 0.2


color_img_path = dir_path + 'JPEGImages/'
label_img_path = dir_path + 'Segmentations/'

imgs = os.listdir(color_img_path)
limgs = os.listdir(label_img_path)


imgs.sort(key=str.lower)
limgs.sort(key=str.lower)

f1 = open('data/train1.txt', 'w')
f2 = open('data/val1.txt', 'w')

pd = tqdm(total=len(imgs))

for img, limg in zip(imgs, limgs):
    pd.update(1)
    if np.random.random() > val_size:
        f1.write(color_img_path + img + ' ' + label_img_path + limg + '\n')
    else:
        f2.write(color_img_path + img + ' ' + label_img_path + limg + '\n')

pd.close()
f1.close()
f2.close()
