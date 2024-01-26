
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os
import dataframe_image as dfi
import numpy as np
from natsort import natsorted
from sklearn.metrics import confusion_matrix, classification_report


def final_table_info(base_log_dir, class_n):
    files_starting_with_events_out = [f for f in os.listdir(base_log_dir) if f.startswith('events.out')]
    sorted_filenames = sorted(files_starting_with_events_out, key=lambda x: int(x.split('.')[-1]))
    print(sorted_filenames)

    all_data = pd.DataFrame()
    for i in range(len(sorted_filenames)):
        print(sorted_filenames[i])
        log_dir = os.path.join(base_log_dir, sorted_filenames[i])  # Correct path joining here
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload() 

        # Get scalar data
        scalar_tags = event_acc.Tags()['scalars']
        print(scalar_tags)
        tags = [
            'final/test_class_0_auc', 'final/test_class_1_auc', 'final/test_class_2_auc',
            'final/test_class_0_acc', 'final/test_class_1_acc', 'final/test_class_2_acc',
            'final/val_error', 'final/val_overall_auc', 'final/test_error', 'final/test_overall_auc'
        ]
        # Extracting data for each tag and storing in dictionary
        data = {}
        for tag in tags:

            if tag in scalar_tags:
                scalar_events = event_acc.Scalars(tag)
                data[tag] = scalar_events[-1].value if scalar_events else None
            else:
                data[tag] = None
            
        current_df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
        current_df.columns = ["Metric", f"Fold_{[i]}"]  
        if all_data.empty:
            all_data = current_df
        else:
            all_data = pd.merge(all_data, current_df, on="Metric", how="outer")



    print(all_data)

    styled_df = all_data.style.set_table_styles(
        [{'selector': 'th', 'props': [('background-color', 'white'), ('color', 'black'),('padding', '10px')]},
        {'selector': 'td', 'props': [('border', '1px solid black')]}]
    ).set_properties(**{'text-align': 'center'}).background_gradient(cmap='coolwarm', low=0.5, high=0.5,vmin= 0.8)

    dfi.export(styled_df, base_log_dir+'/df_styled.png')
    return all_data

def classify_string(s):
    if "Sub_Benign" in s or "Main_3_class" in s:
        return 3
    else:
        return 2


if __name__ == "__main__":
    #options
    option_1 = "first_level" #[first_level, second_level, final_level]
    class_n = 3


    
    final_level_dir = '/home/mlam/Documents/Research_Project/WSI_domain/Output/OUTPUT_WSI_domain_specific_v2/'
    second_level_dir = 'CLAM-MB_64_lookahead_adamw_True_500/'
    first_level_dir = 'exp_3_0/' #exp_"X"_0/ where X is the number of fold

    if option_1 == "first_level":
        base_dir = final_level_dir + second_level_dir + first_level_dir
        print(base_dir)
        pred_labels = np.load(base_dir + "pred_labels.npy")
        labels = np.load(base_dir + "labels.npy")
        probs = np.load(base_dir + "probs.npy")
        cm = confusion_matrix(labels, pred_labels)
        print(cm)

        report = classification_report(labels, pred_labels)
        print(report)

    elif option_1 == "second_level":
        base_dir = final_level_dir + second_level_dir
    elif option_1 == "final_level":
        base_dir = final_level_dir

        directories_only = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        keywords = ["CLAM-MB", "adamw"]
        directories_only = [directory for directory in directories_only if all(keyword in directory for keyword in keywords)]
        print (directories_only)
        
        directories_only = natsorted(directories_only)
        print (directories_only)
        #directories_only = [item for item in directories_only if "False" not in item]


        df2 = pd.DataFrame()
        print(len(directories_only))
        for i in range (len(directories_only)):
            directory = base_dir + directories_only[i]
            class_n = classify_string(directories_only[i])
            print(directory)
            df = final_table_info(directory, class_n)
            df.replace(to_replace=[None], value=np.nan, inplace=True)
            mean_series = df.iloc[:, 1:].mean(axis=1)
            std_series = df.iloc[:, 1:].std(axis=1)
            df2["Metric"] = df["Metric"]
            df2[directories_only[i] + " Mean"] = mean_series
            #df2[directories_only[i] + " Std Deviation"] = std_series
            print(df2)
        
        styled_df = df2.style.background_gradient(cmap='coolwarm', low=0.5, high=0.5, axis=None, vmin=0.8)
        styled_df.to_html(base_dir+'styled_dataframe_False.html')

    