"""
Adapted from CLAM: https://github.com/mahmoodlab/CLAM
"""
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

class Bag_Dataset(Dataset):
    """
    Dataset enumerating over slides

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self,csv_path):
        """_summary_

        Args:
            csv_path (str): path to slides
        """
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) :
        return self.df['slide_id'][idx]


class Instance_Dataset(Dataset):
    """
    Dataset enumerating patches of a slide

    Args:
        Dataset (s): _description_
    """
    def __init__(self,wsi,slide_path,slide_id) -> None:
        self.wsi = wsi
        self.slide_id = slide_id
        self.patch_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))]
        )
        self.slide_path = slide_path
     
        with h5py.File(self.slide_path,'r') as f:
            patch_dataset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(patch_dataset)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
    
        with h5py.File(self.slide_path,'r') as f:
            coord = f['coords'][idx]
            coord = torch.from_numpy(coord)
        #print(coord, self.patch_level, (self.patch_size, self.patch_size), type(coord), type(self.patch_level), (self.patch_size, self.patch_size))
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.patch_transform(img).unsqueeze(0)
        return img, coord, self.slide_id

class Instance_Dataset_heatmap(Dataset):
    def __init__(self,wsi,coords,patch_level, patch_size, slide_id) -> None:
        self.wsi = wsi
        self.slide_id = slide_id
        self.patch_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
 
        coord = self.coords[idx]
        #print(coord, self.patch_level, (self.patch_size, self.patch_size), type(coord), type(self.patch_level), (self.patch_size, self.patch_size))
           
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.patch_transform(img).unsqueeze(0)
        return img, coord, self.slide_id

class Instance_Dataset_heatmap_greyscale_output_channel_1(Dataset):
    def __init__(self, wsi, coords, patch_level, patch_size, reduce_factor=1) -> None:
        self.wsi = wsi
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((patch_size // reduce_factor, patch_size // reduce_factor)),  # Resize the image
            transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale 
        ])
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.reduce_factor = reduce_factor  # New attribute to adjust size reduction

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.patch_transform(img).unsqueeze(0)
        return img, coord



class Instance_Dataset_greyscale_output_channel_1(Dataset):
    """
    Dataset enumerating patches of a slide

    Args:
        Dataset (s): _description_
    """
    def __init__(self, wsi, slide_path, slide_id) -> None:
        self.wsi = wsi
        self.slide_id = slide_id
        # Add Grayscale transformation here
        self.patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
                transforms.Normalize(mean = [(0.485 + 0.456 + 0.406) / 3], std = [(0.229 + 0.224 + 0.225) / 3])  
            ]
        )
        self.slide_path = slide_path
        with h5py.File(self.slide_path, 'r') as f:
            patch_dataset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(patch_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.slide_path, 'r') as f:
            coord = f['coords'][idx]
            coord = torch.from_numpy(coord)
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.patch_transform(img).unsqueeze(0)
        return img, coord, self.slide_id


class Instance_Dataset_greyscale_output_channel_3(Dataset):
    """
    Dataset enumerating patches of a slide

    Args:
        Dataset (s): _description_
    """
    def __init__(self, wsi, slide_path, slide_id) -> None:
        self.wsi = wsi
        self.slide_id = slide_id
        # Add Grayscale transformation here
        self.patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))  # Update the mean and std for single channel
            ]
        )
        self.slide_path = slide_path
        with h5py.File(self.slide_path, 'r') as f:
            patch_dataset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(patch_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.slide_path, 'r') as f:
            coord = f['coords'][idx]
            coord = torch.from_numpy(coord)
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.patch_transform(img).unsqueeze(0)
        return img, coord, self.slide_id