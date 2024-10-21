# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:45:00 2024

@author: n1kraz
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This functions can be updated to include reference to the generated merged_datasets
# Then they will be independent of importing merged datasets in the file





def plot_dataset_columns(dataset_merged):
    """This function makes a plot that combines all available parameters in one figure,
    alligns them on time scale and allows to see similarity of the patterns"""

    print('\nTrain dataset\n\n')
    # Plot each column
    i = 1
    plt.figure(figsize=(7*10/6, 7*5/6))
    for group in range(dataset_merged.shape[1]):
        plt.subplot(dataset_merged.shape[1], 1, i)
        plt.plot(dataset_merged.values[:, group])
        plt.title(dataset_merged.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()


def plot_dataset_corr_heatmap(dataset_merged, dataset_name):
    # Visualize the correlation matrix using a heatmap
    corr_matrix_tot  = dataset_merged.corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_tot, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title(dataset_name)
    plt.show()


def corr(dataset_merged, par1, par2, v_line):
    # Extract two time series
    x = dataset_merged.dropna()[par1].values
    y = dataset_merged.dropna()[par2].values
    
    
    # Compute cross-correlation
    corr = np.correlate(x, y, mode='full') 
    
    #Normalize the cross correlation
    corr/=np.max(corr)
    
    # Plot results
    # Create a plot of the cross-correlation values
    lags = np.arange(-len(x)+1, len(y))
    plt.plot(lags, corr)
    plt.axvline(x = v_line, color = 'r')
    plt.axhline(y = 1, color = 'r')
    plt.xlabel('Lag')
    plt.ylabel('Normalized Cross-correlation')
    plt.title(f'Cross-correlation between {par1} and {par2}')
    plt.show()