"""
Dataloader for Features slide level
"""
import os
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset


class Feature_bag_dataset(Dataset):
    """
    Dataloader for Features at slide level
    """
    def __init__(self,root, csv_path, split_path = False, fold_num = None, split = None, num_classes=3, class_name="", augment=False, result_dir='') -> None:
        """_summary_

        Args:
            root (str): root path
            csv_path (str): path to csv file
            split (str, optional): Split train, val or test. Defaults to None.
            num_classes (int, optional): number of classes. Defaults to 5.
        """
        super(Feature_bag_dataset,self).__init__()
        df = pd.read_csv(csv_path)
        label_dict = {}
        if class_name != "":
            if class_name == 'Hyperplasia':
                df = df.loc[df['Label'] == class_name]
                label_dict = {'atypical_hyperplasia':0, 'hyperplasia_no_atypia':1}
            elif class_name == 'Neoplasia':
                df = df.loc[df['Label'] == class_name]
                label_dict = {'endometrioid_adenocarcinoma':0, 'serous_carcinoma':1}
            elif class_name == 'Benign':
                df = df.loc[df['Label'] == class_name]
                label_dict = {'atrophic_endometrium':0, 'proliferative_endometrium':1, 'secretory_endometrium': 2}
            elif class_name == 'all':
                label_dict = {'atrophic_endometrium':0,
                              'proliferative_endometrium':1,
                              'secretory_endometrium': 2,
                              'atypical_hyperplasia':3,
                              'hyperplasia_no_atypia':4,
                              'endometrioid_adenocarcinoma':5,
                              'serous_carcinoma':6
                              }
            
        df = df[[ "slide_id", "Sublabel"]]
        # print(df.head)
        df['Sublabel'] = df['Sublabel'].map(label_dict)
        self.label_dict = label_dict
        self.df = df
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.augment = augment

        self.root = root
        self.split = split
        self.csv_path = csv_path
        self.class_name = class_name
        self.result_dir = result_dir
    
        if split_path:
            self.split_data = self.read_split_csv(split_path, fold_num)
            self.df = self.apply_split(self.split_data, split)
        else:
            self.df = self.split_dataset()

        self.num_classes = num_classes
        self.cls_slide_id_prep()

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        if '_' in str(self.df['slide_id'][idx]):
            feature_path = os.path.join(self.result_dir, 'augmented_images',  str(self.df['slide_id'][idx]) + '.hdf5')
            with h5py.File(feature_path, "r") as f:
                # print('Keys')
                # print(list(f.keys()))
                features = torch.from_numpy(f["patches"][()])
        else:
            path_slide = os.path.join(self.root, str(self.df['slide_id'][idx]))
            features = torch.concat([torch.load(os.path.join(path_slide,file), map_location=torch.device('cpu'))['features'] for file in os.listdir(path_slide)])
        # print('features')
        # print(features)
        # print('features type')
        # print(type(features))

        return features, torch.tensor(self.df['Sublabel'][idx]), str(self.df['slide_id'][idx])


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
            final_df = self.df[self.df['slide_id'].isin(filter_ids)].reset_index(drop=True)
            if split_type == 'train' and self.augment:
                aug_df = pd.read_csv(os.path.join(self.result_dir, self.csv_path.split('/')[-1].split('.')[0] + '_augmented.csv'))
                aug_df = aug_df[aug_df['Label'] == self.class_name]
                aug_df = aug_df[[ "slide_id", "Sublabel"]]
                aug_df = aug_df[aug_df['slide_id'].str.contains("_")].reset_index(drop=True)
                aug_df['Sublabel'] = aug_df['Sublabel'].map(self.label_dict)
                # print("Aug_df")
                # print(aug_df)
                # print("df")
                # print(final_df)
                # aug_df = aug_df.astype({'slide_id': pd.StringDtype()})
                # final_df = final_df.astype({'slide_id': pd.StringDtype()})
                final_df = pd.concat((final_df, aug_df)).reset_index(drop=True)
                final_df = final_df.astype({'slide_id': pd.StringDtype()})
                # print("final_df")
                # print(final_df)
            return final_df
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
            self.slide_cls_ids[i] = np.where(self.df['Sublabel'] == i)[0]
    
    def getlabel(self, ids):
        return self.df['Sublabel'][ids]



if __name__ == "__main__":
    # Define the root and CSV file paths
    root_path = "/home/mlam/Documents/Research_Project/images_data/Output_clam_grey_images/FEATURES_DIRECTORY_BW_256_v3__KimiaNet_greyscale_True_pretrained_output_ch_1/images/"  # Change this to your actual data directory
    csv_path = "/home/mlam/Documents/Research_Project/images_data/Output_clam_grey_images/FEATURES_DIRECTORY_BW_256_v3__KimiaNet_greyscale_True_pretrained_output_ch_1/filtered_images_clean.csv"  # Change this to your actual csv file path
    split_path = "/home/mlam/Documents/Research_Project/images_data/Output_clam_grey_images/splits/task_2_tumor_subtyping_100_all/"

  
    # Create dataset instances
    train_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split_path=split_path, fold_num=0, split="train")
    val_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split_path=split_path, fold_num=0, split="val")
    test_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split_path=split_path, fold_num=0, split="test")

    # Example usage
    print(train_dataset[0][0].shape)
    print(len(train_dataset))

    train_dataset = Feature_bag_dataset(root=root_path, csv_path=csv_path, split="train")
    print(train_dataset[0][0].shape)