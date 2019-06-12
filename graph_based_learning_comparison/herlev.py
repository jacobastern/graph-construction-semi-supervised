import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision
from IPython import display

import shutil
import zipfile

# assert torch.cuda.is_available() # You need to request a GPU from Runtime > Change Runtime Type

import os
from IPython.core.ultratb import AutoFormattedTB
__ITB__ = AutoFormattedTB(mode = 'Verbose',color_scheme='LightBg', tb_offset = 1)

import PIL

class HerlevDataset(Dataset):
    def __init__(self, destination, train=True, download=True, transform=None, target_transform=None, size=256, split=.7):
        
        # Location to extract zip
        self.root = os.path.join(destination, 'smear2005')
        # Train/test split
        self.split = split
        
        if download and not os.path.exists(self.root):
            datasets.utils.download_url("http://mde-lab.aegean.gr/images/stories/docs/smear2005.zip", './', "pap_smear_data.zip", None)
            self.extract_zip("pap_smear_data.zip")
            os.remove('pap_smear_data.zip')
        
        self.clean_data()
        self.organize_directories()
        
        if transform is None and target_transform is None:
            transform = transforms.Compose([
                            transforms.RandomRotation(360),
                            transforms.RandomAffine(0, translate=(.05, .05), resample=PIL.Image.BILINEAR),
                            transforms.Resize((size, size)), 
                            transforms.ToTensor()])
            target_transform = transforms.Compose([
                            transforms.Resize((size, size)), 
                            transforms.ToTensor()])
        else:
            transform = transform
            target_transform = target_transform
        
        # Note that these transforms are applied when we sample from the dataset, not when it is created
        if train:
            self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(self.root, "train_images"), 
                        transform=transform)
        else:
            self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(self.root, "test_images"), 
                        transform=target_transform)
        
        self.abnormal_classes = [0, 1, 2, 6]

    def organize_directories(self):
        """Splits data into train and test directories"""
        
        # Original directory of all images
        search_directory = os.path.join(self.root, "New database pictures")
        
        # Create train/test/validation folders
        train_dir = os.path.join(self.root, "train_images")
        test_dir = os.path.join(self.root, "test_images")
                
        for root, dirs, files in os.walk(os.path.normpath(search_directory), topdown=False):

            for f in files:
                
                # Get the name of the containing folder
                base_dir = os.path.basename(root)
    
                # Replicate directory structure
                if not os.path.exists(os.path.join(train_dir, base_dir)):
                    os.makedirs(os.path.join(train_dir, base_dir))
                if not os.path.exists(os.path.join(test_dir, base_dir)):
                    os.makedirs(os.path.join(test_dir, base_dir))
                
                # Move files
                num = np.random.rand(1)
                if num > self.split:
                    shutil.move(os.path.join(search_directory, base_dir, f), os.path.join(test_dir, base_dir))
                else:
                    shutil.move(os.path.join(search_directory, base_dir, f), os.path.join(train_dir, base_dir))
   
    def clean_data(self):
        """Removes undesired images."""
        
        # Original directory of all images
        search_directory = os.path.join(self.root, "New database pictures")
        cnt = 0
        # Delete all but raw images
        for root, dirs, files in os.walk(os.path.normpath(search_directory), topdown=False):
            for name in files:
                if not name.endswith('.BMP') or "-d" in name:
                    source_file = os.path.join(root, name)
                    os.remove(source_file)
                    
    def extract_zip(self, zip_path):
        print('Unzipping {}'.format(zip_path))
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))
            
    def __getitem__(self, index):
        """Returns:
            img (torch.Tensor): the image
            class (bool): True if abnormal, False if normal
        """
        img, img_class = self.dataset_folder[index]
        return img, img_class in self.abnormal_classes
    
    def __len__(self):
        return len(self.dataset_folder)
    
def show(img):
    """Displays the image in line

    Parameters:
        img (1 x 3 x a x b Tensor): An image in the Pytorch Tensor format
    """
    img = img.squeeze(0).permute(1,2,0)
    img = img.clamp(0, 1)
    plt.imshow(img, interpolation='nearest')
    plt.grid(False)
    plt.axis('off')
    
def show_dataset(dataset, n=6):
    img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
    plt.imshow(img)
    plt.axis('off')