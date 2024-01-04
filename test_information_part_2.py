
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os
import dataframe_image as dfi



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



if __name__ == "__main__":
    base_log_dir = '/home/mlam/Documents/Research_Project/WSI_domain/Output/CLAM_MB_csv_cross_2024_01_03/'
    class_n = 3
    base_base_dir = '/home/mlam/Documents/Research_Project/WSI_domain/Output/OUTPUT_WSI_domain_specific/'
    directories_only = [f for f in os.listdir(base_base_dir) if os.path.isdir(os.path.join(base_base_dir, f))]
    print (directories_only)
    for i in range (len(directories_only)):
        directory = base_base_dir + directories_only[i]
        print(directory)
        final_table_info(directory, class_n)
        