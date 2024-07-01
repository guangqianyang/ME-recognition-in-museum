import torch
import torch.nn as nn
import cv2
import os
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



def image_loader(image_path, image_size):
    img1 = cv2.imread(os.path.join(image_path, '0.png'),0)
    img2 = cv2.imread(os.path.join(image_path, '1.png'),0)
    img3 = cv2.imread(os.path.join(image_path, '3.png'),0)

    # resize 
    img1 = cv2.resize(img1, (image_size, image_size))
    img2 = cv2.resize(img2, (image_size, image_size))
    img3 = cv2.resize(img3, (image_size, image_size))

    img1 = np.expand_dims(img1, axis=2)
    img2 = np.expand_dims(img2, axis=2)
    img3 = np.expand_dims(img3, axis=2)

    img1 = np.array(img1, np.float32).transpose(2, 0, 1) / 255.0 
    img2 = np.array(img2, np.float32).transpose(2, 0, 1) / 255.0 
    img3 = np.array(img3, np.float32).transpose(2, 0, 1) / 255.0 

    return img1, img2, img3



class Imageloader(Dataset):
    def __init__(self, image_size, image_root, imglist,data_type):
        self.image_size = image_size
        self.image_root = image_root
        self.image_list = imglist
        if data_type=='MME':
            self.data = pd.read_excel('annotations/annotation_MME.xlsx')
        elif data_type=='realMME':
            self.data = pd.read_excel('annotations/annotation_realMME.xlsx')
        elif data_type=='LabMME':
            self.data = pd.read_excel('annotations/annotation_LabMME.xlsx')
        self.data = self.data.set_index('name')
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        image_path = os.path.join(self.image_root, name)
        #print(image_path)
        label = self.data.loc[name]['category']
        assert label in ['others', 'negative', 'positive','surprise']
        if label == 'others':
            value = 0
        elif label == 'negative':
            value = 1
        elif label == 'positive':
            value = 2
        else:
            value = 3
        image1, image2, image3 = image_loader(image_path, self.image_size)
        return image1, image2, image3, value

class Imageloader_test(Dataset):
    def __init__(self, image_size, image_root, annotation):
        self.image_size = image_size
        self.image_root = image_root
        self.image_list = os.listdir(image_root)
        print(self.image_list)
        self.data = pd.read_excel(annotation)
        self.data.sort_values(by=['name'], inplace=True)
        self.data = self.data.set_index('name')
        #print(self.data)

    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        image_path = os.path.join(self.image_root, name)
        #print(image_path)
        #print('name is: ', name)
        label = self.data.loc[name]['category']

        if label == 'others':
            value = 0
        elif label == 'negative':
            value = 1
        elif label == 'positive':
            value = 2
        else:
            value = 3
        image1, image2, image3 = image_loader(image_path, self.image_size)
        return image1, image2, image3, value


