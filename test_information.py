
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import pandas as pd



def plot_tensorboard_scalars(event_acc, train_tag, val_tag, title, x_label, y_label):
    """
    Plots scalar values from TensorBoard logs given specific tags for training and validation.

    Args:
    event_acc (EventAccumulator): The loaded TensorBoard event accumulator.
    train_tag (str): Tag for training data.
    val_tag (str): Tag for validation data.
    title (str): The title of the plot.
    x_label (str): The label for the X-axis.
    y_label (str): The label for the Y-axis.
    """
    # Extract the event data for training and validation
    train_events = event_acc.Scalars(train_tag)
    val_events = event_acc.Scalars(val_tag)

    # Process the data into lists of steps and values
    train_steps = [e.step for e in train_events]
    train_values = [e.value for e in train_events]

    val_steps = [e.step for e in val_events]
    val_values = [e.value for e in val_events]

    # Initialize your figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))  

    # Plot training and validation data
    ax.plot(train_steps, train_values, label=f'Train {train_tag}')
    ax.plot(val_steps, val_values, label=f'Validation {val_tag}')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()

def single_plot_tensorboard_scalars(event_acc, train_tag,  title, x_label, y_label):

    train_events = event_acc.Scalars(train_tag)

    train_steps = [e.step for e in train_events]
    train_values = [e.value for e in train_events]


    # Initialize your figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))  

    # Plot training and validation data
    ax.plot(train_steps, train_values, label=f'{train_tag}')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()




# Path to your tensorboard log directory
log_dir = '/home/mlam/Documents/Research_Project/WSI_domain/Output/CLAM_MB_seed_2024_01_03/events.out.tfevents.1704277495.txubtune.122915.0'
class_n = 3
# Initialize an event accumulator
event_acc = EventAccumulator(log_dir)
event_acc.Reload()  # Load all the data written so far

# Get scalar data
scalar_tags = event_acc.Tags()['scalars']


# Define the tags you want to include in the table
tags = [
    'final/test_class_0_auc', 'final/test_class_1_auc', 'final/test_class_2_auc',
    'final/test_class_0_acc', 'final/test_class_1_acc', 'final/test_class_2_acc',
    'final/val_error', 'final/val_overall_auc', 'final/test_error', 'final/test_overall_auc'
]

# Extracting data for each tag and storing in dictionary
data = {}
for tag in tags:
    scalar_events = event_acc.Scalars(tag)
    # Assuming there's only one event per tag, or you want the last one
    if scalar_events:
        data[tag] = scalar_events[-1].value  # Gets the last value, you might want to adjust this

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])

# Show the table
print(df, '\n')


print(scalar_tags)

#ACCURACY --------------------------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(10, 15))  # Adjust the size as needed

for i, class_num in enumerate(range(class_n)):
    # Train accuracy for each class
    train_acc = event_acc.Scalars(f'train/class_{class_num}_acc')
    val_acc = event_acc.Scalars(f'val/class_{class_num}_acc')

    # Extracting step and value for plotting
    train_steps = [e.step for e in train_acc]
    train_values = [e.value for e in train_acc]

    val_steps = [e.step for e in val_acc]
    val_values = [e.value for e in val_acc]

    # Plotting train and val accuracy on the same subplot for each class
    ax[i].plot(train_steps, train_values, label=f'Train Class {class_num} Accuracy')
    ax[i].plot(val_steps, val_values, label=f'Validation Class {class_num} Accuracy')
    ax[i].set_title(f'Class {class_num} Accuracy')
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Accuracy')
    ax[i].legend()

# Display the plot
plt.tight_layout(pad=3.0, h_pad=3.0) 
plt.show()






#LOSS --------------------------------------------------------------------------------------
plot_tensorboard_scalars(event_acc, 'train/loss', 'val/loss', 'Training & Validation Loss', 'Epoch', 'Loss')
plot_tensorboard_scalars(event_acc, 'train/error', 'val/error', 'Training & Validation Error', 'Epoch', 'Error')

single_plot_tensorboard_scalars(event_acc,  'val/auc', 'Validation AUC', 'Epoch', 'AUC')
single_plot_tensorboard_scalars(event_acc,  'train/clustering_loss', 'Train Clustering Loss', 'Epoch', 'Loss')
single_plot_tensorboard_scalars(event_acc,  'val/inst_loss', 'Validation Inst Loss', 'Epoch', 'Loss')

