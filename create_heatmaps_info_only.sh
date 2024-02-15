
# Config
# set the configs

data_path=/home/mlam/Documents/Research_Project/images_data/IMAGES-Copy/ALL_images/
patch_path=/home/mlam/Documents/Research_Project/images_data/Output/RESULTS_DIRECTORY_BW_256_v3/
feature_path=/home/mlam/Documents/Research_Project/images_data/Output/FEATURES_DIRECTORY_BW_256_v3_1__KimiaNet_greyscale_True_pretrained_output_ch_1/images/
#heatmap_dir=/home/mlam/Documents/Research_Project/WSI_domain/Output/OUTPUT_WSI_domain_specific_v2/CLAM-MB_64_lookahead_adamw_True_500/exp_3_0/Heatmap_test/
#ckpt_path=/home/mlam/Documents/Research_Project/WSI_domain/Output/OUTPUT_WSI_domain_specific_v2/CLAM-MB_64_lookahead_adamw_True_500/exp_3_0/model.pt

heatmap_dir=/home/mlam/Documents/Research_Project/WSI_domain/Output/CLAM_MB_dummy_2/exp_0_0/Heatmap_test/
ckpt_path=/home/mlam/Documents/Research_Project/WSI_domain/Output/CLAM_MB_dummy_2/exp_0_0/model.pt

# ["ResNet","KimiaNet", "DenseNet"]
feature_ext="KimiaNet" 
# ["CLAM-SB", "CLAM-MB", "TransMIL"]
model="CLAM-MB"


python3 heatmaps_info.py \
  --heatmap_dir $heatmap_dir \
  --feat_dir $feature_path \
  --slide_dir $data_path \
  --csv_path $patch_path/process_list_autogen.csv \
  --gpu=True \
  --n_classes 3 \
  --ckpt_path $ckpt_path \
  --model $model \
  --feature_ext $feature_ext \
  --drop_out \
  --Main_or_Sub_label Main_3_class \
  --k_sample_CLAM 64
