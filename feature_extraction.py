"""
Fast script to extraction of features
"""
import pytorch_lightning as pl
from  dataset import Bag_Dataset,Instance_Dataset, Instance_Dataset_greyscale_output_channel_3, Instance_Dataset_greyscale_output_channel_1
from models.feature_model import Feature_extract
import argparse
import os
import openslide
from torch.utils.data import DataLoader, ConcatDataset
from utils import collate_features, save_hdf5
import writer 
import torch
import h5py

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--model', type=str, choices = ["ResNet","KimiaNet", "DenseNet"],default="KimiaNet")
parser.add_argument('--devices', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--num_workers',type=int,default=2)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--strategy', type=str, default="ddp")
parser.add_argument('--greyscale', type=bool, default=True)
parser.add_argument('--pretrained_output_channel', type=int, choices = [1,3], default=3) # [1,3]
parser.add_argument('--start_id', type=str, default=None) # [1,3]
args = parser.parse_args()

if __name__ == "__main__":
    
    bags = Bag_Dataset(args.csv_path)
    all_dataloaders = []
    flag = False
    feat_dir = args.feat_dir +'_'+  str(args.model) + '_greyscale_' + str(args.greyscale) + '_pretrained_output_ch_' + str(args.pretrained_output_channel)
    os.makedirs(feat_dir, exist_ok=True)
    for bag_idx in range(len(bags)):
        slide_id = bags[bag_idx].split(args.slide_ext)[0]

        if not os.path.isfile(os.path.join(args.data_slide_dir, slide_id+args.slide_ext)):
            slide_id += '.svs'  #

        
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        wsi = openslide.open_slide(slide_file_path)
        if args.greyscale == True and args.pretrained_output_channel == 3:
            patches_dataset = Instance_Dataset_greyscale_output_channel_3(wsi, h5_file_path,slide_id)
        if args.greyscale == True and args.pretrained_output_channel == 1:
            patches_dataset = Instance_Dataset_greyscale_output_channel_1(wsi, h5_file_path,slide_id)
        else:
            patches_dataset = Instance_Dataset(wsi,h5_file_path,slide_id)

        kwargs = {'num_workers': args.num_workers} if torch.cuda.is_available() else {}
        patch_loader = DataLoader(dataset = patches_dataset, batch_size=args.batch_size, **kwargs,collate_fn=collate_features)
        if slide_id == str(args.start_id) or None == args.start_id:
            flag = True
        if flag == True:
            print(slide_id)
            all_dataloaders.append(patch_loader)

    pred_writer = writer.PredWriter(feat_dir)
    trainer = pl.Trainer(accelerator="gpu",devices = args.devices,callbacks=pred_writer,strategy=args.strategy)
    model = Feature_extract(model=args.model, pretrained_output_channel=args.pretrained_output_channel)
    prediction = trainer.predict(model,all_dataloaders)        
