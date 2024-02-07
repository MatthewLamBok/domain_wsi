"""
Dataloader for Features slide level
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset
from utils import *

class Feature_bag_dataset(Dataset):
    """
    Dataloader for Features at slide level
    """
    def __init__(self, root, csv_path, split_path = False, Main_or_Sub_label = 'Main_3_class', fold_num = None, split = None, num_classes=3) -> None:
        """_summary_

        Args:
            root (str): root path
            csv_path (str): path to csv file
            split (str, optional): Split train, val or test. Defaults to None.
            num_classes (int, optional): number of classes. Defaults to 5.
        """
        super(Feature_bag_dataset,self).__init__()
        
        df = filter_dataframe(csv_path, Main_or_Sub_label)
        
        class_counts = df["Label"].value_counts()
        print(class_counts)


        print("Dataframe Shape :", df.shape)            
        self.df = df
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.root = root
        self.split = split
        
    
        if split_path:
            self.split_data = self.read_split_csv(split_path, fold_num)
            self.df = self.apply_split(self.split_data, split)
        else:
            self.df = self.split_dataset()
            print("Dataframe Shape split :", self.df.shape)

        self.num_classes = num_classes
        self.cls_slide_id_prep()

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        path_slide = os.path.join(self.root, str(self.df['slide_id'][idx]))
        features = torch.concat([torch.load(os.path.join(path_slide,file), map_location=torch.device('cpu'))['features'] for file in os.listdir(path_slide)])
        #print(path_slide, self.df['slide_id'][idx], self.df['Label'][idx], features.shape)
        return features, torch.tensor(self.df['Label'][idx])


    def split_dataset(self):
        pat_ids = self.df.slide_id.unique()
        train_ids, rem_ids = train_test_split(pat_ids, test_size=0.4)
        val_ids, test_ids = train_test_split(rem_ids, test_size=0.5)
        train_df = self.df.loc[self.df['slide_id'].isin(train_ids)].reset_index(drop=True)
        val_df = self.df.loc[self.df['slide_id'].isin(val_ids)].reset_index(drop=True)
        test_df = self.df.loc[self.df['slide_id'].isin(test_ids)].reset_index(drop=True)
        if self.split == 'train':
            return train_df
        elif self.split == 'val':
            return val_df
        elif self.split == 'test':
            return test_df
        else:
            raise NotImplementedError

    def read_split_csv(self, split_path, fold_num):
        split_df = pd.read_csv(split_path + 'splits_' + str(fold_num) + '.csv')
        return {
            'train': split_df['train'].dropna().astype(int).tolist(),
            'val': split_df['val'].dropna().astype(int).tolist(),
            'test': split_df['test'].dropna().astype(int).tolist()
        }

    def apply_split(self, split_data, split_type):
        # Filter main dataframe based on the IDs from the split data
        if split_type in split_data:
            filter_ids = split_data[split_type]
            return self.df[self.df['slide_id'].isin(filter_ids)].reset_index(drop=True)
        else:
            raise ValueError(f"Split type {split_type} not found in split data")


    def split_dataset_cross_val_no_csv(self, num_folds, current_fold):
        pat_ids = self.df.slide_id.unique()
        train_val_ids, test_ids = train_test_split(pat_ids, test_size=0.2)
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Splitting the dataset for cross-validation
        splits = list(kf.split(train_val_ids))  # Creating all splits
        train_ids, val_ids = train_val_ids[splits[current_fold][0]], train_val_ids[splits[current_fold][1]]

        # Creating dataframes for each set based on current fold
        train_df = self.df[self.df['slide_id'].isin(train_ids)].reset_index(drop=True)
        val_df = self.df[self.df['slide_id'].isin(val_ids)].reset_index(drop=True)
        test_df = self.df[self.df['slide_id'].isin(test_ids)].reset_index(drop=True)

        # Return the respective dataframe based on the split argument
        if self.split == 'train':
            return train_df
        elif self.split == 'val':
            return val_df
        elif self.split == 'test':
            return test_df
        else:
            raise ValueError(f"Split type {self.split} not recognized or implemented")


    def cls_slide_id_prep(self):
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.df['Label'] == i)[0]
    
    def getlabel(self, ids):
        return self.df['Label'][ids]

def filter_dataframe(csv_path, Main_or_Sub_label):
    df = pd.read_csv(csv_path)
    df = df[["slide_id","Label_2class", "Label", "Sublabel"]]

    if Main_or_Sub_label == 'Main_3_class':
        print("0_", Main_or_Sub_label)
        df = df[["slide_id", "Label"]]
        label_dict = {'Benign':0, 'Hyperplasia':1, 'Neoplasia':2}
        df['Label'] = df['Label'].map(label_dict)
    elif Main_or_Sub_label == 'Main_2_class':
        print("1_", Main_or_Sub_label)
        df = df[["slide_id", "Label_2class"]]
        label_dict = {'Benign':0, 'Cancer':1}
        df['Label_2class'] = df['Label_2class'].map(label_dict)
        df = df.rename(columns={"slide_id": "slide_id", "Label_2class": "Label"})
    elif Main_or_Sub_label == 'Sub_Benign':
        print("2_", Main_or_Sub_label)
        df = df[df['Label'] == 'Benign'][['slide_id', 'Sublabel']]
        label_dict = {'atrophic_endometrium':0, 'secretory_endometrium':1, 'proliferative_endometrium':2}
        df['Sublabel'] = df['Sublabel'].map(label_dict)
        df = df.rename(columns={"slide_id": "slide_id", "Sublabel": "Label"})
    elif Main_or_Sub_label == 'Sub_Hyperplasia':
        print("3_", Main_or_Sub_label)
        df = df[df['Label'] == 'Hyperplasia'][['slide_id', 'Sublabel']]
        label_dict = {'atypical_hyperplasia':0, 'hyperplasia_no_atypia':1}
        df['Sublabel'] = df['Sublabel'].map(label_dict)
        df = df.rename(columns={"slide_id": "slide_id", "Sublabel": "Label"})
    elif Main_or_Sub_label == 'Sub_Neoplasia':
        print("4_", Main_or_Sub_label)
        df = df[df['Label'] == 'Neoplasia'][['slide_id', 'Sublabel']]
        label_dict = {'endometrioid_adenocarcinoma':0, 'serous_carcinoma':1}
        df['Sublabel'] = df['Sublabel'].map(label_dict)
        df = df.rename(columns={"slide_id": "slide_id", "Sublabel": "Label"})
    else: 
        print("Error >>>>>")
        exit()
    return df

if __name__ == "__main__":
    # Define the root and CSV file paths
    root_path = "/home/mlam/Documents/Research_Project/images_data/Output/FEATURES_DIRECTORY_BW_256_v3_1__KimiaNet_greyscale_True_pretrained_output_ch_1/images/"  # Change this to your actual data directory
    csv_path = "/home/mlam/Documents/Research_Project/images_data/Output/FEATURES_DIRECTORY_BW_256_v3_1__KimiaNet_greyscale_True_pretrained_output_ch_1/filtered_images_clean.csv"  # Change this to your actual csv file path
    split_path = "/home/mlam/Documents/Research_Project/images_data/Output_clam_grey_images/splits/task_2_tumor_subtyping_100_all/"

    
    #filter_dataframe(csv_path=csv_path, Main_or_Sub_label = 'Sub_Neoplasia')
    
    # Create dataset instances
    train_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split_path=split_path, Main_or_Sub_label = 'Sub_Neoplasia', fold_num=0, split="train",num_classes=2)
    weights = make_weights_for_balanced_classes_split(train_dataset)
    exit()
    val_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split_path=split_path, Main_or_Sub_label = 'Main_3_class', fold_num=0, split="val")
    test_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split_path=split_path, Main_or_Sub_label = 'Main_3_class',  fold_num=0, split="test")

    # Example usage
    print(train_dataset[0][0].shape)
    print(len(train_dataset))

    train_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split="train")
    print(train_dataset[0][0].shape)

