"""
Adapted from CLAM: https://github.com/mahmoodlab/CLAM
"""
import argparse
from  dataset import Bag_Dataset,Instance_Dataset, Instance_Dataset_heatmap
import pytorch_lightning as pl
import os
import openslide
import torch
from torch.utils.data import DataLoader
from utils import *
import feature_dataset 
import math
from heatmap_utils import *
import csv
import math
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import histomicstk as htk

import numpy as np
import scipy as sp
from statistics import mean 
import skimage.io
import skimage.measure
import skimage.color
from skimage.measure import regionprops, label
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import squidpy as sq


def infer_one_slide(args, model, device, features):
    """_summary_

    Args:
        args (argparse.Namespace): Arguments for the system
        model (torr.nn.Module): Model to infer
        device (torch.device): Device for infernce
        features (torch.Tensor): Features of the slide

    Returns:
        A (torch.Tensor): Heatmap
    """
    model.eval()
    features = features.to(device)
    with torch.no_grad():
        if args.model == "CLAM-SB" or args.model== "CLAM-MB":
            logits, Y_prob, Y_hat, A, _ = model(features, coord_attn=torch.ones((features.shape[0])))
            Y_hat = Y_hat.item()
            if args.model == "CLAM-MB":
                A = A[Y_hat]
            A = A.view(-1, 1)
        elif args.model == "TransMIL":
            logits, Y_prob, Y_hat, return_dict = model(data = features.unsqueeze(0), return_attn=True)
            n = features.shape[0]
            add_length = (math.ceil(math.sqrt(n)))**2 -n
            n2 = n + add_length +1
            padding = 256 - (n2%256) if n2%256 > 0 else 0
            A = return_dict['A'][:,:,padding:(padding+n+1),padding:(padding+n+1)][:,:,0,:-1].view(8,-1,1)
        
    return A.cpu().numpy(), Y_hat

def histogram_info(Attention_score):
    print(Attention_score.shape)
    #Normalization between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Attention_score = scaler.fit_transform(Attention_score)
    
    binsizes = math.sqrt(len(Attention_score))
    plt.hist(Attention_score, bins=int(binsizes), edgecolor='black')
    plt.title('Histogram of Your Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def stardist_2D_versatile_he(img, model,  nms_thresh=None, prob_thresh=None):
    # axis_norm = (0,1)   # normalize channels independently
    axis_norm = (0, 1, 2)  # normalize channels jointly
    # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
    # this is the default normalizer noted in StarDist examples.
    img = normalize(img, 1, 99.8, axis=axis_norm)
    
    labels, _ = model.predict_instances(
        img, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    return labels

def part_3_test(data_dict, slide_id, display_bool):

    slide_file_path='/home/mlam/Documents/Research_Project/images_data/IMAGES-Copy/ALL_images/'+slide_id+'.svs'
    h5_file_path_arg = '/home/mlam/Documents/Research_Project/images_data/Output/RESULTS_DIRECTORY_BW_256_v3/'
    h5_file_path =os.path.join(h5_file_path_arg, 'patches', slide_id+'.h5')
    sorted_indices = np.argsort(data_dict[slide_id]['attention_score'], axis=0).flatten()
    sorted_attention_score = data_dict[slide_id]['attention_score'][sorted_indices]
    sorted_coords = data_dict[slide_id]['coords'][sorted_indices]

    top_10_attention_scores = sorted_attention_score[:10]
    sorted_coords = sorted_coords[:10]

    
    wsi = openslide.open_slide(slide_file_path)
  
    patches_dataset = Instance_Dataset_heatmap(wsi=wsi,coords=sorted_coords, patch_level= 0, patch_size = 256, slide_id=slide_id)
    print(torch.max(patches_dataset[0][0]),torch.min(patches_dataset[0][0]))


    model = StarDist2D.from_pretrained("2D_versatile_he")
    num_cells=[]    
    for i in range(len(sorted_coords)):
        crop = patches_dataset[i][0].squeeze().permute(1, 2, 0).numpy()
        if display_bool:
            crop_squid = sq.im.ImageContainer(crop)
            sq.im.segment(
                img=crop_squid,
                layer="image",
                model= model,
                channel=None,
                method=stardist_2D_versatile_he,
                layer_added="segmented_stardist_default",
                prob_thresh=0.3,
                nms_thresh=None,
            )
            print(
                f"Number of segments in crop: {len(np.unique(crop_squid['segmented_stardist_default']))}"
            )
            fig, axes = plt.subplots(1, 2)
            crop_squid.show("image", ax=axes[0])
            _ = axes[0].set_title("H&H")
            crop_squid.show("segmented_stardist_default", cmap="jet", interpolation="none", ax=axes[1])
            _ = axes[1].set_title("segmentation")
            plt.show()
        
        labels = stardist_2D_versatile_he(crop, model= model, nms_thresh=None, prob_thresh=0.3)
        props = regionprops(labels)
        cell_sizes = [prop.area for prop in props] 
        print(len(props)) 
        num_cells.append(len(props)) 
    
    
    print(sum(num_cells))

def display_patches_grid(dataset, indices, title_prefix, max_cols=5):
    total_patches = len(indices)
    ncols = min(total_patches, max_cols)
    nrows = math.ceil(total_patches / ncols)  # Calculate rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
    if nrows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Ensure axes is iterable for a single row
    
    for idx, ax in zip(indices, axes):
        img = dataset[idx][0].squeeze().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f'{title_prefix} {idx+1}')
        ax.axis('off')
    for ax in axes[total_patches:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def part_4_test(data_dict, slide_id, display_bool):

    slide_file_path='/home/mlam/Documents/Research_Project/images_data/IMAGES-Copy/ALL_images/'+slide_id+'.svs'
    h5_file_path_arg = '/home/mlam/Documents/Research_Project/images_data/Output/RESULTS_DIRECTORY_BW_256_v3/'
    h5_file_path =os.path.join(h5_file_path_arg, 'patches', slide_id+'.h5')
    sorted_indices = np.argsort(data_dict[slide_id]['attention_score'], axis=0).flatten()
    sorted_attention_score = data_dict[slide_id]['attention_score'][sorted_indices]
    sorted_coords = data_dict[slide_id]['coords'][sorted_indices]

    digit= 8
    top_10_attention_scores = sorted_attention_score[:digit]
    top_sorted_coords = sorted_coords[:digit]
    low_10_attention_scores = sorted_attention_score[digit:]
    low_sorted_coords = sorted_coords[digit:]
    
    wsi = openslide.open_slide(slide_file_path)
  
    patches_dataset = Instance_Dataset_heatmap(wsi=wsi,coords=sorted_coords, patch_level= 0, patch_size = 256, slide_id=slide_id)
    print(torch.max(patches_dataset[0][0]),torch.min(patches_dataset[0][0]))
    
    coords, percent_array = feature_dataset.coord_weight(sorted_coords, slide_file_path, 0.8, display_bool=False)
    print(coords, percent_array)
    top_indices = range(digit)  # Top 'digit' patches
    low_indices = range(digit, digit + digit)  # Next 'digit' number of patches after the top ones

    # Display top patches
    display_patches_grid(patches_dataset, top_indices, "Top Attention", max_cols=5)
    # Display low attention patches
    # Adjust 'low_indices' as needed based on your specific requirements
    display_patches_grid(patches_dataset, low_indices, "Low Attention", max_cols=5)



def info_stats_measure(Attention, slide_id):
    attention_scores_dict_stats = {}
    scores_array = np.array(Attention)
    
    # Calculating various statistics
    attention_score_mean = scores_array.mean()
    attention_score_std = scores_array.std()
    attention_score_median = np.median(scores_array)
    attention_score_range = scores_array.ptp()  # Peak to peak (max - min)
    attention_score_skewness = stats.skew(scores_array)
    attention_score_kurtosis = stats.kurtosis(scores_array)
    attention_score_iqr = np.percentile(scores_array, 75) - np.percentile(scores_array, 25)  # Interquartile range

    # Adding the statistics to the dictionary
    attention_scores_dict_stats = {
        'attention_score_mean': attention_score_mean,
        'attention_score_std': attention_score_std,
        'attention_score_median': attention_score_median,
        'attention_score_range': attention_score_range,
        'attention_score_skewness': attention_score_skewness[0],
        'attention_score_kurtosis': attention_score_kurtosis[0],
        'attention_score_iqr': attention_score_iqr
    }
    return attention_scores_dict_stats

parser = argparse.ArgumentParser("Heatmap Inference script")
parser.add_argument("--model", type=str,choices=["CLAM-SB","CLAM-MB","TransMIL"],default="CLAM-SB")
parser.add_argument("--feature_ext",type=str,choices=["ResNet","KimiaNet", "DenseNet"],default="ResNet")
parser.add_argument("--ckpt_path",type=str, required=True)
parser.add_argument("--heatmap_dir",type=str,required=True)
parser.add_argument("--feat_dir",type=str,required=True)
parser.add_argument("--slide_dir",type=str,required=True)
parser.add_argument("--csv_path",type=str,default=None)
parser.add_argument("--gpu",type=bool,default=False)
parser.add_argument("--slide_ext",type=str,default=".svs")
parser.add_argument("--instance_loss",type=str,default="svm")
parser.add_argument("--n_classes",type=int,required=True)
parser.add_argument('--drop_out',action="store_true",default=False)
parser.add_argument("--vis_level",type=int, default=-1)
parser.add_argument("--alpha", type=float, default=0.4)
parser.add_argument("--blank_canvas", action="store_true", default=False)
parser.add_argument("--use_ref_scores", action="store_true", default=False)
parser.add_argument("--blur",action="store_true",default=False)
parser.add_argument("--binarize",action="store_true",default=False)
parser.add_argument("--binary_thresh",type=int, default=1)
parser.add_argument("--custom_downsample",type=int,default=1)
parser.add_argument('--k_sample_CLAM', type=int, default=8)
parser.add_argument("--Main_or_Sub_label", type=str, choices=["Main_3_class","Main_2_class","Sub_Benign","Sub_Hyperplasia","Sub_Neoplasia"],default="Main_3_class")

#addition for CLAM

args = parser.parse_args()
if __name__=="__main__":

    #parameter=1
    correlation_bool = False
    data_dict = {}
    attention_scores_dict_stats = {}

    device=torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = create_model(args,device)
    print("HERE", args.ckpt_path)
    print(list(torch.load(args.ckpt_path,map_location=torch.device("cpu")).keys()))
    model.load_state_dict(torch.load(args.ckpt_path,map_location=torch.device("cpu")))

    model.to(device)
    os.makedirs(args.heatmap_dir,exist_ok=True)
    heatmaps_vis_args = heatmap_vis_args = {'convert_to_percentiles': not args.use_ref_scores, 'vis_level': args.vis_level, 'blur': args.blur, 'custom_downsample': args.custom_downsample}
    


    df = feature_dataset.filter_dataframe(args.feat_dir+"../filtered_images_clean.csv", args.Main_or_Sub_label)

    for count, dir in enumerate(os.listdir(args.slide_dir)):
        slide_id = dir.split('.')[0]
        if os.path.isdir(os.path.join(args.feat_dir,slide_id)):
            os.makedirs(os.path.join(args.heatmap_dir,slide_id),exist_ok=True)

            path = os.path.join(args.feat_dir,slide_id)
            feature = torch.concat([torch.load(os.path.join(path,file), map_location=torch.device('cpu'))['features'] for file in os.listdir(path)])
            coords = torch.concat([torch.tensor(torch.load(os.path.join(path,file), map_location=torch.device('cpu'))['coords']) for file in os.listdir(path)]).numpy()
            A, Y_hat = infer_one_slide(args,model,device,feature)

            Y = df[df['slide_id'] == int(slide_id)]['Label'].iloc[0]
            print(f"Slide id {slide_id} Predicted: {Y_hat} Actual: {Y}")

            #histogram_info(Attention_score = A)

            if Y_hat == Y:
                data_dict[slide_id] = {'attention_score': A, 'coords': coords, 'Y_hat': Y_hat, 'Y': Y}
                attention_scores_dict_stats[slide_id] = info_stats_measure(Attention = A, slide_id= slide_id)
                attention_scores_dict_stats[slide_id]['Label'] = Y_hat
                if Y_hat != 0 and count > 0:
                    part_4_test(data_dict, slide_id=slide_id, display_bool= True)

        else:
            print("NO DIRECTORY FOUND SKIPPING")

        if count == 3000:
            break

 


    if correlation_bool == True:
        df = pd.DataFrame.from_dict(attention_scores_dict_stats, orient='index')

        # Creating the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='viridis')  # You can choose different colormaps like 'coolwarm', 'Blues', etc.
        plt.title('Attention Scores Heatmap')
        plt.xlabel('Statistics')
        plt.ylabel('Slide ID')
        plt.show()
    
    