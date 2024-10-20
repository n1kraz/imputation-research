#!/usr/bin/env python
# coding: utf-8

# # Machine learning model for data imputation

# ## Imports



# Libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Mathematical operations and scientific computations
import math

# For saving and loading objects
import os
import sys
import pickle
import configparser
import argparse
from pathlib import Path

# Change working directory to the location of this script
script_directory = Path(__file__).resolve().parent
os.chdir(script_directory)


# Scaling input dataset samples to increase model performance
from sklearn.preprocessing import MinMaxScaler

# Deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import keras_tuner as kt

# Display TensorFlow version
print("TensorFlow version: ", tf.__version__, "\n")

# Settings for displaying numpy arrays
np.set_printoptions(suppress=True, precision=3)

# Settings for displaying large pandas DataFrames
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Setting home dirrectory for the project
home_dir = Path.cwd().parent
print('Home dirrectory of the project:', home_dir)
sys.path.append(os.path.join(home_dir, 'checks'))   # The path to the folder containing modules
# Setting path to the data
data_path = f"{home_dir}\data"





print("TensorFlow built with CUDA support:", tf.test.is_built_with_cuda())




from tensorflow.python.client import device_lib
device_lib.list_local_devices()




# Allocating only as much memory as needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ## Dataset import



# Interval at which data were resampled
interval = '30s'
# Number of the padding values added at the beginning and end of every event in the dataset
max_possible_lag_size = 90
max_possible_gap_size = 90




# Load the dictionary containing normalized datasets
with open(f'{home_dir}\data\prepared_datasets\multipar_datasets_{interval}_norm.pkl', 'rb') as f:
    multipar_datasets_norm = pickle.load(f)

# Access individual datasets
test_dataset_norm = multipar_datasets_norm['test_datasets']
val_dataset_norm = multipar_datasets_norm['val_datasets']
train_dataset_norm = multipar_datasets_norm['train_datasets']


# ## Dataset import


def main(config_file):
    # Create a ConfigParser object
    configuration = configparser.ConfigParser()

    # Read the configuration file
    configuration.read(config_file)

    # Access model architecture type and trget parameter from the config file
    target = configuration['Model']['target']
    architecture = configuration['Model']['architecture']

    # Access settings of this specific test case
    lag_size  = int(configuration['Test_case']['lag_size'])
    gap_size  = int(configuration['Test_case']['gap_size'])
    batch_size  = int(configuration['Test_case']['batch_size'])
    
    # Access settings of optimization process
    n_trials  = int(configuration['Optimization']['n_trials'])
    n_epochs_trial  = int(configuration['Optimization']['n_epochs_trial'])

    # Access number of training epochs for the optimized model
    n_epochs = int(configuration['Training']['n_epochs'])
    


    return {
        'architecture': architecture,
        'target': target,
        'lag_size': lag_size,
        'gap_size': gap_size,
        'batch_size': batch_size,
        'n_trials': n_trials,
        'n_epochs_trial': n_epochs_trial,
        'n_epochs': n_epochs
    }


# Importing configuration
if __name__ == "__main__":
    # config_file = 'configs/config.ini'
    # config = load_config(config_file)
    parser = argparse.ArgumentParser(description="Run script with a configuration")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    args = parser.parse_args()
    config = main(args.config)


# # Model

# ## Windowing technique


def find_gap_indices(arr):
    gap_info = []
    for col in arr.T:  # Iterate over each column
        gap_starts = []
        gap_ends = []
        gap_lengths = []
        in_gap = False
        gap_start = None
        for i, value in enumerate(col):
            if np.isnan(value):
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    in_gap = False
                    gap_starts.append(gap_start)
                    gap_ends.append(i - 1)
                    gap_lengths.append(i - gap_start)
        if in_gap:
            # If the last gap continues till the end of the column
            gap_starts.append(gap_start)
            gap_ends.append(len(col) - 1)
            gap_lengths.append(len(col) - gap_start)
        gap_info.append((gap_starts, gap_ends, gap_lengths))
    return gap_info


def fill_gap_linspace(arr):
    gap_info_array = find_gap_indices(arr)
    for par in range(arr.shape[1]):
        for gap in range(len(gap_info_array[par][0])):
            st = gap_info_array[par][0][gap]
            end = gap_info_array[par][1][gap]
            length = gap_info_array[par][2][gap]
            arr[st:end+1, par] = np.linspace(arr[st-1, par], arr[end+1, par], length+2)[1:-1]
    return arr

# This function returns one of the parameters padded with -1s
# Likely would not work correctly with CNNs and MLPs
def pad_window_with_mask(window, par_col, lag_size, gap_size):
     # Rows to pad
    indices = tf.range(lag_size, lag_size+gap_size)
    # Convert updates to the same data type as the input tensor
    updates = tf.cast(tf.fill(indices.shape, -1), window.dtype)
    # Updating tensor
    new_window = tf.tensor_scatter_nd_update(window, tf.transpose([indices, tf.fill(indices.shape, par_col)]), updates)
    return new_window

def pad_window_with_interp(window, par_col, lag_size, gap_size):
    # Convert window to a tensor if it's not already one
    window_tensor = tf.convert_to_tensor(window, dtype=tf.float32)

    # Rows to pad
    indices = tf.range(lag_size, lag_size + gap_size)

    # Generate the updates with the correct length
    start_value = window_tensor[lag_size-1, par_col]
    end_value = window_tensor[lag_size + gap_size, par_col]
    updates = tf.linspace(start_value, end_value, gap_size+2)[1:-1]

    # Creating the indices tensor for tensor_scatter_nd_update
    scatter_indices = tf.stack([indices, tf.fill([gap_size], par_col)], axis=1)

    # Updating tensor
    new_window = tf.tensor_scatter_nd_update(window_tensor, scatter_indices, updates)
    return new_window

def windowed_dataset(dataset_list, lag_size, max_possible_lag_size, gap_size, batch_size, par_col, shuffle=False):
    """
    Generates windows from the dataset
    
    Args:
      dataset_list (list of arrays) - contains arrays of values for each event
      lag_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      predefined_lag_size (int) - when dataset was created, there were added tails of constant size to each event, this is their length
      shuffle (bool) - switcher for shuffling the dataset

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
  
    # Initialize an empty dataset to store all concatenated datasets
    merged_dataset = None
    
    for event in dataset_list:
        # Dataset is extended with tails of some maximum duration, now we need to cut off parts of the tails, to leave only the lags of proper length
        # I also shift the tails by +/-1, to allow having at least one datapoint at the first and last instance for the parameter context
        # extended_event = event[max_possible_lag_size - lag_size + 1: -(max_possible_lag_size - lag_size + 1)]

        # I will try instead to run model without any preceding lag, because padded tails might confuse the model during training
        extended_event = event[max_possible_lag_size: -(max_possible_lag_size)]
        

        # Ensuring all data are float
        extended_event = extended_event.astype('float32')
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(extended_event)
        
        # Window the data but only take those with the specified size
        dataset = dataset.window(2 * lag_size + gap_size, shift=1, drop_remainder=True)
    
        # Flatten the windows by putting its elements in a single window
        dataset = dataset.flat_map(lambda window: window.batch(2 * lag_size + gap_size))

        # Padding dataset with linear iteration
        dataset = dataset.map(lambda window: (pad_window_with_interp(window, par_col, lag_size, gap_size),  tf.expand_dims(window[:, par_col], axis=-1)), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Concatenate the current dataset with the final_dataset
        if merged_dataset is None:
            merged_dataset = dataset
        else:
            merged_dataset = merged_dataset.concatenate(dataset)
    
    # Caching data (pipeline optimization step)
    merged_dataset = merged_dataset.cache()
    
    if shuffle == True:
        # buffer_size should be higher than the biggest dataset
        buffer_size = math.ceil((len(np.concatenate(dataset_list)))/1000)*1000
        # Shuffle the windows
        merged_dataset = merged_dataset.shuffle(buffer_size)
    
    # Create batches of windows
    merged_dataset = merged_dataset.batch(batch_size)
    
    # Prefetching dataset (pipeline optimization step)
    merged_dataset = merged_dataset.prefetch(tf.data.AUTOTUNE)
    
    return merged_dataset


# ## Preparing dataset


# Directory for the models
filepath = f"{home_dir}/models_archive/{config['target']}_models/interpolation_tuned_{config['architecture']}-{config['lag_size']}-{config['gap_size']}-{interval}"

# Column indeces for the features in the original dataset
initial_param_columns = {'cond': 0, 'temp': 1, 'DO': 2, 'turb': 3, '-pH': 4, 'flow': 5, 'cum_disch':6, 'time_since_last_rain':7, 'time_since_rain_started':8, 'absolute_gradient_flow':9}
# initial_param_columns = {'pollution'  :0, 'dew' : 1, 'temp' : 2, 'press' : 3, 'wnd_spd' : 4, 'snow' : 5, 'rain' : 6}

# Chossing parameters that will be part of the model (feature and target selection)
if config['target'] == 'turb':
    feature_target_param = ['turb', 'flow', 'time_since_rain_started']
elif config['target'] == '-pH':
    feature_target_param = ['-pH', 'flow', 'time_since_rain_started']
elif config['target'] == 'cond':
    feature_target_param = ['cond', 'flow', 'time_since_rain_started']

# Feature selection
train_dataset_norm_1 = [event[:, [initial_param_columns[param] for param in feature_target_param]] for event in train_dataset_norm]
val_dataset_norm_1 = [event[:, [initial_param_columns[param] for param in feature_target_param]] for event in val_dataset_norm]
test_dataset_norm_1 = [event[:, [initial_param_columns[param] for param in feature_target_param]] for event in test_dataset_norm]
train_dataset_norm = train_dataset_norm_1
val_dataset_norm = val_dataset_norm_1
test_dataset_norm = test_dataset_norm_1

# Selecting right corresponding columns
new_param_columns = {key: index for index, key in enumerate(feature_target_param)}
par_col = new_param_columns[config['target']]


# <br><br><br>*Windowing dataset*


train_dataset_windowed = windowed_dataset(train_dataset_norm, config['lag_size'], max_possible_lag_size, config['gap_size'], config['batch_size'], par_col=par_col, shuffle=True)
val_dataset_windowed = windowed_dataset(train_dataset_norm, config['lag_size'], max_possible_lag_size, config['gap_size'], config['batch_size'], par_col=par_col)

n_batches = len(list(train_dataset_windowed.as_numpy_iterator()))
print('\nNumber of batches:', n_batches)


# ## Running the model


def find_variable_name(variable):
    for name, value in globals().items():
        if value is variable:
            return name

def save_history(history, directory, filename):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Join the directory and filename
    filepath = os.path.join(directory, filename)

    # Save the history to a file
    with open(filepath, 'wb') as file:
        pickle.dump(history, file)


# <br><br><br>*Model names*



def make_model_name(hps, arch):
    if arch == 'CNN':
        model_name = ''
        model_name += 'h1_1'
        model_name += str(hps.get('CNN_head1_1'))
        model_name += 'h1_2'
        model_name += str(hps.get('CNN_head1_2'))
        model_name += '_ks'
        model_name += str(hps.get('kern_size_head1'))
        model_name += '-'
        model_name += 'h2_1'
        model_name += str(hps.get('CNN_head2_1'))
        model_name += 'h2_2'
        model_name += str(hps.get('CNN_head2_2'))
        model_name += '_ks'
        model_name += str(hps.get('kern_size_head2'))
        model_name += '-'
        model_name += 'h3_1'
        model_name += str(hps.get('CNN_head3_1'))
        model_name += 'h3_2'
        model_name += str(hps.get('CNN_head3_2'))
        model_name += '_ks'
        model_name += str(hps.get('kern_size_head3'))
        model_name += '-'
        model_name += 'd_n'
        model_name += str(hps.get(f'Dense_layer_n_units'))

    elif arch == 'LSTM':
        model_name = ''
        model_name += 'lstm_1_'
        model_name += str(hps.get(f'Bi_LSTM_0'))
        model_name += '-'
        model_name += 'lstm_2_'
        model_name += str(hps.get(f'Bi_LSTM_1'))
        model_name += '-'
        model_name += 'td_d_'
        model_name += str(hps.get(f'TD_Dense_layer_n_units'))

    elif arch == 'MLP':
        model_name = ''
        model_name += 'l1_'
        model_name += str(hps.get('MLP_head1'))
        model_name += '-l2_'
        model_name += str(hps.get('MLP_head2'))
        model_name += '-'
        model_name += 'd_n'
        model_name += str(hps.get(f'Dense_layer_n_units'))
    return model_name


# ### Optymization algorithm

# <br><br><br>***Three-headed CNN***



if config['architecture'] == 'MLP':
    def model_builder(hp):
        # Define input layer
        number_of_features = len(feature_target_param)
        input_shape = (2 * config['lag_size'] + config['gap_size'], number_of_features)
        inputs = layers.Input(shape=input_shape)
        
        # Add a Masking layer to handle the -111.0 values
        masked_inputs = layers.Masking(mask_value=-111.0, dtype = 'float32')(inputs)
        
        # Split the input into three individual features along the last axis
        feature_1 = layers.Lambda(lambda x: x[:, :, 0])(masked_inputs)
        feature_2 = layers.Lambda(lambda x: x[:, :, 1])(masked_inputs)
        feature_3 = layers.Lambda(lambda x: x[:, :, 2])(masked_inputs)
        
        # Reshape features to be compatible with Dense layers
        feature_1 = layers.Flatten()(feature_1)
        feature_2 = layers.Flatten()(feature_2)
        feature_3 = layers.Flatten()(feature_3)
    
    
        layer_1 = layers.Dense(units = hp.Int("MLP_head1", min_value=128, max_value=2048, step=128))
        layer_2 = layers.Dense(units = hp.Int("MLP_head2", min_value=128, max_value=2048, step=128))
        drop_1 = layers.Dropout(rate=hp.Float('Dropout_rate_head1', min_value=0.1, max_value=0.5))
        drop_2 = layers.Dropout(rate=hp.Float('Dropout_rate_head1', min_value=0.1, max_value=0.5))
            
        # Head 1
        branch_1_1 = layer_1(feature_1)
        drop_1_1 = drop_1(branch_1_1)
        branch_1_2 = layer_2(drop_1_1)
        drop_1_2 = drop_2(branch_1_2)
    
        # Head 2
        branch_2_1 = layer_1(feature_2)
        drop_2_1 = drop_1(branch_2_1)
        branch_2_2 = layer_2(drop_2_1)
        drop_2_2 = drop_2(branch_2_2)
    
        # Head 3
        branch_3_1 = layer_1(feature_3)
        drop_3_1 = drop_1(branch_3_1)
        branch_3_2 = layer_2(drop_3_1)
        drop_3_2 = drop_2(branch_3_2)
    
        # Merge the outputs of the three heads
        combined = layers.concatenate([drop_1_2, drop_2_2, drop_3_2])
    
        # Interpretation layer
        interpreting_layer = layers.Dense(units=hp.Int("Dense_layer_n_units", min_value=512, max_value=2048, step=512), activation='relu')(combined)
        drop_dense = layers.Dropout(rate=hp.Float('Dropout_rate_last', min_value=0.1, max_value=0.4))(interpreting_layer)
        
        # Output layer
        output = layers.Dense(2 * config['lag_size'] + config['gap_size'])(drop_dense)
    
        # Defining the model
        model = Model(inputs=inputs, outputs=output)
        
        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Float('learning_rate', 5e-8, 1e-5, sampling='log')
        
        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='mse',
                      metrics=['mean_absolute_error', 'mean_squared_error'])
    
        return model

elif config['architecture'] == 'CNN':
    # Define the model builder function
    def model_builder(hp):
        # Define input layer
        number_of_features = len(feature_target_param)
        input_shape = (2 * config['lag_size'] + config['gap_size'], number_of_features)
        inputs = layers.Input(shape=input_shape)
    
        # Add a Masking layer to handle the -111.0 values
        masked_inputs = layers.Masking(mask_value=-111.0, dtype='float32')(inputs)
        
        # Function to create a head
        def create_head(hp, head_num, ks_min, ks_max):
            ks = hp.Int(f"kern_size_head{head_num}", min_value=ks_min, max_value=ks_max, step=1)
            conv = layers.Conv1D(
                filters=hp.Int(f"CNN_head{head_num}_1", min_value=128, max_value=1024, step=128),
                kernel_size=ks,
                padding='same')(masked_inputs)
            conv = layers.Conv1D(
                filters=hp.Int(f"CNN_head{head_num}_2", min_value=128, max_value=1024, step=128),
                kernel_size=ks,
                padding='same')(conv)
            drop = layers.Dropout(rate=hp.Float(f'Dropout_rate_head{head_num}', min_value=0.1, max_value=0.5))(conv)
            pool = layers.MaxPooling1D(pool_size=2)(drop)
            return layers.Flatten()(pool)
    
        # Create the three heads with the helper function
        flat1 = create_head(hp, head_num=1, ks_min=2, ks_max=3)
        flat2 = create_head(hp, head_num=2, ks_min=4, ks_max=5)
        flat3 = create_head(hp, head_num=3, ks_min=6, ks_max=7)
    
    
        
        # Merge the outputs of the three heads
        merged = layers.concatenate([flat1, flat2, flat3])
    
        # Interpretation layer
        dense = layers.Dense(units=hp.Int("Dense_layer_n_units", min_value=512, max_value=2048, step=512), activation='relu')(merged)
        drop_dense = layers.Dropout(rate=hp.Float('Dropout_rate_last', min_value=0.1, max_value=0.4))(dense)
        
        # Output layer
        outputs = layers.Dense(2 * config['lag_size'] + config['gap_size'])(drop_dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Float('learning_rate', 5e-8, 1e-5, sampling='log')
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse',
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
    
        return model

elif config['architecture'] == 'LSTM':
    def model_builder(hp):
        # For input layer
        number_of_features = len(feature_target_param)
        
        # Defining model
        model = tf.keras.models.Sequential()
    
        # Input shape
        model.add(tf.keras.layers.Input(shape=(2 * config['lag_size'] + config['gap_size'], number_of_features)))
    
        # Masking tails
        model.add(tf.keras.layers.Masking(mask_value = -111.0, dtype = 'float32'))
        
        # Tune the number of layers.
        for i in range(2):
            model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        # Tune number of units separately.
                        units = hp.Int(f"Bi_LSTM_{i}", min_value = 128, max_value = 1024, step = 128),
                        return_sequences=True
                    )
                )
            )
            model.add(
                tf.keras.layers.Dropout(
                    # Tuning dropout rate individually for each layer
                    rate = hp.Float(f'Dropout_rate_{i}', min_value = 0, max_value = 0.5)
                )
            )
    
        # Interpretation layer
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units = hp.Int(
                        f"TD_Dense_layer_n_units",
                        min_value = 128,
                        max_value = 2048,
                        step = 128
                    )
                )
            )
        )
        model.add(tf.keras.layers.Dropout(rate=hp.Float(f'Dropout_rate_last', min_value=0.1, max_value=0.5)))
        # Dense layer for interpretation
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
    
        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Float('learning_rate', 5e-8, 1e-5, sampling='log')
    
        # Compiling the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate),
            loss = 'mse',
            metrics = ['mean_absolute_error', 'mean_squared_error']
        )
        
        return model




# Use Bayesian Optimization for tuning
tuner = kt.BayesianOptimization(
    model_builder,
    objective = kt.Objective('val_mean_squared_error', direction = 'min'),
    max_trials = config['n_trials'],
    directory = f'{filepath}',
    project_name = 'bayesian_optimization_logs'
)

# Running the search
tuner.search(
    train_dataset_windowed,
    epochs = config['n_epochs_trial'],
    validation_data = val_dataset_windowed,
    verbose = 1,
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_mean_squared_error', patience = 10)]
)



# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
# Number of the best trial
print(tuner.oracle.get_best_trials(num_trials=10)[0].trial_id)


# ### Training of the model with optimal hyperparameters


print('Best configuration:', make_model_name(best_hps, config['architecture']))


# Build the model with the optimal hyperparameters and retrain it
model = tuner.hypermodel.build(best_hps)

# Filepaths 
filepath_model = f"{filepath}/{make_model_name(best_hps, config['architecture'])}"
filepath_history = f"{filepath}/history"

# Fit and save models
history = model.fit(
    train_dataset_windowed,
    validation_data = val_dataset_windowed,
    epochs = config['n_epochs'],
    verbose = 1,
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = filepath_model,
            monitor = 'val_mean_squared_error',
            mode = 'min',
            save_best_only = True,
            verbose = 0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_mean_squared_error',
            patience = 100
        )
    ]
)
# Saving history
save_history(history.history, filepath_history, 'history.pkl')


# ### History of training

# Saving history plots


def plot_series(x, y, format=('-', '-'), start=None, end=None, x_name='Time', y_name='Value', title=None, label=None):
    '''
    Visualizes time series data

    Args:
      x (array of int) - contains indeces
      y (array) - contains the measurements for each time step
      format - line style when plotting the graph
      start - first time step to plot
      end - last time step to plot
    '''
    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))
    
    n = 0
    for x_n, y_n in zip(x, y):
        # Plot the time series data for each of the curves consequently
        plt.plot(x_n, y_n, format[n], label=label[n] if label else None)
        n += 1

    # Legend
    plt.legend()
    
    # Title
    plt.title(title)
    
    # Label the x-axis
    plt.xlabel(x_name)

    # Label the y-axis
    plt.ylabel(y_name)

    # Overlay a grid on the graph
    plt.grid(True)
    
    # For plotting only part of the graph
    if start != None or end != None:
        plt.gca().set_xlim(start, end)

    # Saving to history folder
    plt.savefig(f'{filepath}/history/{y_name}.png', dpi=300)

    # Avoiding printing out the figures
    plt.close()


def model_performance(history, title, par, par_name):   
 # Histories of training and validation scores progress
    train = history[par]
    val = history[f'val_{par}']
    # Number of training epochs
    epochs = range(1, len(train) + 1)
    
    # Plot 
    plot_series((epochs, epochs), (train, val), format=('-', '-'), x_name = 'Epochs', y_name = par_name, label = (f'Train {par_name}', f'Val {par_name}'), title = title)  


# Calling history plot functions
model_performance(history.history, title = 'Training/validation metric', par = 'mean_squared_error', par_name = 'MSE')
model_performance(history.history, title = 'Training/validation metric', par = 'mean_absolute_error', par_name = 'MAE')


# Savaing configuration file to history


def load_and_save_config(config_file):
    # Create a ConfigParser object
    configuration = configparser.ConfigParser()

    # Load the existing configuration file
    configuration.read(config_file)

    # Save the modified config to a new folder
    with open(f"{filepath}/history/config.ini", 'w') as configfile:
        configuration.write(configfile)

# Importing configuration
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script with a configuration")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

    args = parser.parse_args()
    load_and_save_config(args.config)


# ## Forecasting

# <br><br><br>*For reversing predictions back to original scale*


# Load the dictionary containing all merged datasets
with open(f'{data_path}/prepared_datasets/multipar_datasets_merged_{interval}.pkl', 'rb') as f:
    multipar_datasets_merged = pickle.load(f)

# Accessing train dataset
train_datasets_merged_multi = multipar_datasets_merged['train_datasets_merged']

# Initiating scaler
scaler = MinMaxScaler(feature_range=(0,1))
# Fitting scaller to ALL train data, so we have to concat precipitation events in train dataset
scaler = scaler.fit(train_datasets_merged_multi)


# <br><br><br>*Running forecast for every instance and saving results*


def process_and_forecast(dataset_norm, lag_size, max_possible_lag_size, gap_size, batch_size, par_col, scaler, model, initial_param_columns, forecast_type, filepath):
    # Number of instances in each event
    n_inst = [(len(el[max_possible_lag_size: -(max_possible_lag_size)]) - 2 * lag_size) - gap_size + 1 for el in dataset_norm]
    # print(f'Number of instances per event ({forecast_type}):', n_inst)

    # Collection of instances for each event
    list_of_instance_arrays = [windowed_dataset([event], lag_size, max_possible_lag_size, gap_size, batch_size, par_col, shuffle=False) for event in dataset_norm]

    forecast_multi_norm, forecast_multi = [], []
    for X in list_of_instance_arrays:
        # Forecast
        f_m_norm = model.predict(X, batch_size=batch_size, verbose=0)
        # Reversing normalization after prediction
        f_m = np.array([scaler.inverse_transform(np.broadcast_to(instance.reshape(-1, 1), (instance.reshape(-1, 1).shape[0], len(initial_param_columns))))[:, par_col] for instance in f_m_norm])

        # These two models return 2-dimentional output
        if config['architecture'] in ['MLP', 'CNN']:
            f_m_norm = np.expand_dims(f_m_norm, axis=-1)
            f_m = np.expand_dims(f_m, axis=-1)
        
        forecast_multi_norm.append(f_m_norm)
        forecast_multi.append(f_m)

    # Filepath for saving forecasts
    filepath_forecasts = f'{filepath}/forecasts'
    if not os.path.exists(filepath_forecasts):
        os.makedirs(filepath_forecasts)

    # Save normalized and regular forecasts to pickle files
    with open(f'{filepath_forecasts}/forecasts_{forecast_type}_norm.pkl', 'wb') as f:
        pickle.dump([forecast_multi_norm], f)

    with open(f'{filepath_forecasts}/forecasts_{forecast_type}.pkl', 'wb') as f:
        pickle.dump([forecast_multi], f)

# Load saved optimal model
model = tf.keras.models.load_model(filepath_model)

# Call the function for test and train datasets
process_and_forecast(test_dataset_norm, config['lag_size'], max_possible_lag_size, config['gap_size'], config['batch_size'], par_col, scaler, model, initial_param_columns, forecast_type='test', filepath=filepath)
process_and_forecast(val_dataset_norm, config['lag_size'], max_possible_lag_size, config['gap_size'], config['batch_size'], par_col, scaler, model, initial_param_columns, forecast_type='val', filepath=filepath)
process_and_forecast(train_dataset_norm, config['lag_size'], max_possible_lag_size, config['gap_size'], config['batch_size'], par_col, scaler, model, initial_param_columns, forecast_type='train', filepath=filepath)

