"""
Code to evaluation
"""
import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models import TransMIL, clam
from sklearn.metrics import confusion_matrix
from topk.svm import SmoothTop1SVM
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import *



def main(args):
    path = args.feat_dir
    device = torch.device(args.device)
    model = create_model(args, device)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    list_dir = os.listdir(args.feat_dir)
    for i in range (len(list_dir)):
        print(list_dir[i])
        path_slide = os.path.join(args.feat_dir, list_dir[i])
        features = torch.concat([torch.load(os.path.join(path_slide, file), map_location=torch.device('cpu'))['features'] for file in os.listdir(path_slide)])
        outputs = model(features.squeeze(0).to(device), coord_attn = torch.ones((features.squeeze(0).shape[0])).to(device))
        print(outputs[0],outputs[2])

parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument("--device",type=int, default=0)
parser.add_argument("--feat_dir", type=str, required=True)
parser.add_argument("--bag_loss", type=str, default="cross-entropy")
parser.add_argument('--instance_loss', type=str, default="svm")
parser.add_argument('--model_path', type=str,default=None)
parser.add_argument('--model', type=str, choices=["CLAM-SB", "CLAM-MB", "TransMIL"], default="CLAM-MB")
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--k_sample_CLAM', type=int, default=64)
parser.add_argument('--drop_out',action="store_true", default=True)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)