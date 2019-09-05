import os
import random
import shutil

path = '/media/runze/DATA/Dasaset/ISIC'
path_img = '/media/runze/DATA/Dasaset/ISIC/image_all'
path_mask = '/media/runze/DATA/Dasaset/ISIC/mask_all'

def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

img_list = os.listdir(path_img)
random.shuffle(img_list)
img_5folder = func(img_list,519)

j = 0
for folder in img_5folder:
    j = j + 1
    print(j)
    m = 0
    for k in folder:
        m = m +1
        print(m)
        shutil.copy(os.path.join(path_img,k), os.path.join(path,'image{}'.format(j)))
        shutil.copy(os.path.join(path_mask,k.split('.')[0]+'_segmentation.png'), os.path.join(path,'mask{}'.format(j)))
