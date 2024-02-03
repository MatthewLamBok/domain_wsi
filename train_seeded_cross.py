"""
Code for training
"""
import argparse
import csv
import os
import re
import time

import feature_dataset as dataset
import numpy as np
import pandas as pd
import torch
from CustomOptim import *
from models import TransMIL, clam
from topk.svm import SmoothTop1SVM
from torch.utils.data import (DataLoader, SequentialSampler,
                              WeightedRandomSampler)
from torch.utils.tensorboard import SummaryWriter
from utils import *
from eval_utils import *

import wandb
import argparse

def plot_reliability_diagram(confidence, accuracy, n_bins=20):
    """
    Plot a reliability diagram with matplotlib.
    """
    # Bin the confidence levels
    bins = np.linspace(0, 1, n_bins + 1)
    binned_confidence = np.digitize(confidence, bins) - 1

    # Calculate accuracy and average confidence per bin
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    for b in range(n_bins):
        in_bin = binned_confidence == b
        if np.any(in_bin):
            bin_accuracy[b] = accuracy[in_bin].mean()
            bin_confidence[b] = confidence[in_bin].mean()

    # Plotting
    plt.plot(bin_confidence, bin_accuracy, marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray')  # Perfectly calibrated line
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.show()

# Function to print arguments nicely
def print_args(args):
    print("Training Configuration:")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")

def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    os.environ['WANDB_DIR'] = args.log_dir
    wandb.login()
    job_type = str.lower(args.class_name) + '_subclassification'
    with wandb.init(project= args.name, config=vars(args), sync_tensorboard=True, mode='offline', group="default_params", job_type=job_type):
        stime = time.time()
        path = args.feat_dir
        data_csv = args.csv_path
        device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available()  else torch.device("cpu")
        writer = SummaryWriter(args.log_dir)
        loss_fn = torch.nn.CrossEntropyLoss()

        print("args.cross_val:", args.cross_val)
        if args.cross_val == 'True':
            loop = 5 
            model_seeds  = 1        
        elif args.cross_val == 'False':
            loop = args.data_seeds
            model_seeds = args.model_seeds

        for data_seed in range(loop):
            print(args.cross_val)
            seed_numpy(data_seed)
            
            if args.cross_val == 'True':
                train_dataset = dataset.Feature_bag_dataset(root=path, csv_path=data_csv, split_path=args.split_path, fold_num=data_seed, split="train", num_classes = args.n_classes, class_name = args.class_name)
                val_dataset = dataset.Feature_bag_dataset(root=path, csv_path=data_csv, split_path=args.split_path, fold_num=data_seed, split="val", num_classes = args.n_classes, class_name = args.class_name)
                test_dataset = dataset.Feature_bag_dataset(root=path, csv_path=data_csv, split_path=args.split_path, fold_num=data_seed, split="test", num_classes = args.n_classes, class_name = args.class_name)
            elif args.cross_val == 'False':
                train_dataset = dataset.Feature_bag_dataset(root=path, csv_path = data_csv, split = 'train', num_classes = args.n_classes, class_name = args.class_name)
                val_dataset = dataset.Feature_bag_dataset(root=path,csv_path = data_csv, split='val', num_classes = args.n_classes, class_name = args.class_name)
                test_dataset = dataset.Feature_bag_dataset(root=path, csv_path=data_csv, split='test', num_classes = args.n_classes, class_name = args.class_name)
            
            weights = make_weights_for_balanced_classes_split(train_dataset)
            train_dataloader = DataLoader(train_dataset, num_workers=4, sampler = WeightedRandomSampler(weights,len(weights)))  
            val_dataloader = DataLoader(val_dataset, num_workers=4, sampler = SequentialSampler(val_dataset))
            test_dataloader = DataLoader(test_dataset, num_workers=4)

            for model_seed in range(model_seeds):
                print(f"Exp/fold:{data_seed}_{model_seed}")
                seed_torch(model_seed,device)
                print("Feature dim: " + str(test_dataset[0][0].shape[1]))
                model = create_model(args, device,test_dataset[0][0].shape[1])
                model = model.to(device)
                print(model)
                val_error, val_auc, _, _= summary(model, test_dataloader, args.n_classes, device, model_type = args.model)
                print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
                wandb.watch(model, log_freq=100)
                optimizer = create_optimizer(args, model,args.model=="TransMIL")
                result_dir = os.path.join(args.result_dir,'exp_'+str(data_seed)+'_'+str(model_seed))
                os.makedirs(result_dir, exist_ok=True)
                exp_idx = (model_seed+1) + (data_seed*3)
                print("Early stopping")
                print(args.early_stopping)
                
                if args.early_stopping:
                    early_stopping = EarlyStopping(patience = 40, stop_epoch=20, verbose = True)
                else:
                    early_stopping = None
                for epoch in range(args.epochs):
                    if args.model == "CLAM-SB" or args.model == "CLAM-MB":
                        print(f"Starting Training {epoch}")
                        train_loop_clam(epoch,model,train_dataloader,optimizer,n_classes=args.n_classes,bag_weight=args.bag_weight,writer=writer,device=device)
                        print(f"Starting Validation {epoch}")
                        stop = validate_clam(epoch,model,val_dataloader,n_classes=args.n_classes,writer=writer,device=device,early_stopping=early_stopping, results_dir =result_dir)
                    elif args.model == "TransMIL":    
                        print(f"Starting Training {epoch}")
                        train_transmil(epoch,model,train_dataloader,device, optimizer=optimizer,n_classes=args.n_classes, loss_fn=loss_fn, writer=writer)
                        print(f"Starting Validation {epoch}")
                        stop = validate_transmil(epoch, model, val_dataloader, n_classes=args.n_classes, device = device,writer=writer,early_stopping=early_stopping, results_dir=result_dir)
                    else:
                        raise NotImplementedError
                    if stop:
                        break
                    # break
                if args.early_stopping:
                    eval_model = create_model(args, device,test_dataset[0][0].shape[1])
                    eval_model.to(device)
                    eval_model.load_state_dict(torch.load(os.path.join(result_dir, "model.pt")))
                else:
                    torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))
                    eval_model = model
                    eval_model.to(device)

                val_error, val_auc, _, _= summary(eval_model, val_dataloader, args.n_classes, device, model_type = args.model, conf_matrix_path=os.path.join(result_dir, 'val_conf_matrix_'+args.model+'.jpg'), save_pred=result_dir)
                print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

                test_error, test_auc, acc_logger, aucs = summary(eval_model, test_dataloader, args.n_classes, device, model_type=args.model, conf_matrix_path=os.path.join(result_dir, 'test_conf_matrix_'+args.model+'.jpg'), save_pred=result_dir)
                print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

                ground_truth, confidence, ece_meaure = confidence_eval(args, eval_model, test_dataloader, device, n_bins = args.bins)

                
                print(f'ECE measure:{ece_meaure}')
                #diagram.plot(confidence, ground_truth, filename=os.path.join(result_dir,"diagram.jpg"))
                
                for i in range(args.n_classes):
                    if len(aucs) > 0:
                        print('class {}: auc: {}'.format(i,aucs[i]))

                        if writer and aucs[i] is not None:
                            writer.add_scalar('final/test_class_{}_auc'.format(i), aucs[i], exp_idx)

                for i in range(args.n_classes):
                    acc, correct, count = acc_logger.get_summary(i)
                    print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

                    if writer and acc is not None:
                        writer.add_scalar('final/test_class_{}_acc'.format(i), acc, exp_idx)

                if writer:
                    writer.add_scalar('final/val_error', val_error, exp_idx)
                    writer.add_scalar('final/val_overall_auc', val_auc, exp_idx)
                    writer.add_scalar('final/test_error', test_error, exp_idx)
                    writer.add_scalar('final/test_overall_auc', test_auc, exp_idx)
                    writer.close()
            # break
               
        end_time = time.time()
        print(f"Time taken: {end_time-stime}")

parser = argparse.ArgumentParser("Training model")
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument("--feat_dir", type=str, required=True)
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--feature_model", type=str, choices=["ResNet","KimiaNet","DenseNet","efficientnet_b0","efficientnet_b1","efficientnet_b2","efficientnet_b3","efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7",'efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l','convnext_tiny','convnext_small','convnext_base','convnext_large', "convunext"],default="KimiaNet")
parser.add_argument("--model", type=str, choices=["CLAM-SB","CLAM-MB","TransMIL"],default="CLAM-SB")
#CLAM
parser.add_argument("--bag_loss", type=str, default="cross-entropy")
parser.add_argument('--instance_loss', type=str, default="svm")
parser.add_argument('--k_sample_CLAM', type=int, default=8)

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument("--bag_weight", type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--opt', type=str, default="lookahead_adamw")
parser.add_argument("--early_stopping", action='store_true', default=False)
parser.add_argument("--result_dir", type=str, default=None)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--drop_out',action="store_true", default=False)
parser.add_argument("--bins", type=int, default=20)

#data_seed/ model seed
parser.add_argument("--data_seeds", type=int, default=5)
parser.add_argument("--model_seeds", type=int, default=3)
#Cross Validation
parser.add_argument("--split_path", type=str, default=None) 
parser.add_argument("--cross_val", type=str, default="False", choices=["False","True"])

# Subclassification
parser.add_argument("--class_name", type=str, default="all")

args = parser.parse_args()

if __name__ == "__main__":
    print_args(args)
    main(args)