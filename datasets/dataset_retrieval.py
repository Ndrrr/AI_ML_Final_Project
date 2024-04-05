import torch 
import torch.nn
from PIL import Image
from torchvision import transforms as tf
import os
import numpy as np
from torch.utils.data import Dataset



class custom_dataset(Dataset):
    def __init__(self, mode = "train", root = "datasets/traffic_data", transforms = None):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transforms = transforms
        
        #select split
        self.folder = os.path.join(self.root, self.mode)
        
        #initialize lists
        self.image_list = []
        self.label_list = []
        
        self.train_aug_transforms = tf.Compose([
                tf.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

                tf.RandomRotation(degrees=15),

                tf.ColorJitter(),

                tf.RandomHorizontalFlip(),

                tf.CenterCrop(size=224),  # Image net standards

                tf.ToTensor(),

                tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        #save class lists
        self.class_list = os.listdir(self.folder)
        self.class_list.sort()
        print(self.class_list)
        
        for class_id in range(len(self.class_list)):
            for image in os.listdir(os.path.join(self.folder, self.class_list[class_id])):
                self.image_list.append(os.path.join(self.folder, self.class_list[class_id], image))
                label = np.zeros(len(self.class_list))
                label[class_id] = 1.0
                self.label_list.append(label)

        # print(self.label_list)
        
    def __getitem__(self, index):
        augmented = index // len(self.image_list)
        index = index % len(self.image_list)

        image_name = self.image_list[index]
        label = self.label_list[index]
        
        
        image = Image.open(image_name)
        if augmented == 1 and self.mode == 'train':
            image = self.train_aug_transforms(image)
        elif(self.transforms):
            image = self.transforms(image)

        label = torch.tensor(label)
        
        return image, label
            
    def __len__(self):
        if (self.mode == 'train'):
            return len(self.image_list) * 2 # adding data augmentation
        return len(self.image_list)