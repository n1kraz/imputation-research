# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:42:54 2023

@author: n1kraz
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setting home dirrectory for the project
home_dir = Path.cwd().parent

# =============================================================================
# Sampling data
# =============================================================================

# Samples collection time

sampling_data_path = f"{home_dir}\data\sampling_data\sample_analysis.xlsx"

def sd_time(date_cell, number_of_samples):
    sd_time = pd.read_excel(sampling_data_path, 
                          sheet_name="BB sample collection", 
                          header = date_cell, 
                          usecols = [0, 2, 4], 
                          nrows = number_of_samples,
                          index_col = "Sample")
    # Proper data types
    sd_time["Start"] = pd.to_datetime(sd_time["Start"])
    sd_time["End"] = pd.to_datetime(sd_time["End"])
    return sd_time



# Laboratory analysis data


def sd_data(first_sample_row, number_of_rows, TSS_first_sample_row, TSS_number_of_rows):
    # Number_of_rows is a total number of rows with data for that field trip (number of samples * number of multiple measurements with lab instrument)
    # Calling Excel table, assigning the row where the first sample data is and assigning number of samples (look in excel table)
    sd_data = pd.read_excel(sampling_data_path, 
                          sheet_name="BB analysis", 
                          header = first_sample_row - 2, 
                          usecols = [0, 2, 3, 5, 6], 
                          nrows = number_of_rows,
                          index_col = "Unnamed: 0").dropna()
    
    # Renaming column titles
    # sd_data = sd_data.rename(columns = {sd_data.columns[0]:"P_Conductivity"})
    sd_data = sd_data.rename(columns = {sd_data.columns[0]:"P_Turbidity"})
    sd_data = sd_data.rename(columns = {sd_data.columns[1]:"L_Conductivity"})
    sd_data = sd_data.rename(columns = {sd_data.columns[2]:"L_Turbidity"})
    sd_data = sd_data.rename(columns = {sd_data.columns[3]:"L_pH"})
    
    sd_TSS = pd.read_excel(sampling_data_path, 
                       sheet_name="BB TSS and LOI", 
                       header = TSS_first_sample_row - 2, 
                       usecols = [0, 8, 10], 
                       nrows = TSS_number_of_rows,
                       index_col = "Unnamed: 0").dropna()
    
    # Renaming column titles
    sd_TSS = sd_TSS.rename(columns = {sd_TSS.columns[0]:"TSS"})
    sd_TSS = sd_TSS.rename(columns = {sd_TSS.columns[1]:"LOI"})
    

    # Index data type is integer
    sd_data.index = sd_data.index.astype(int)
    # Turning zeros from Excel table to NaNs
    sd_data[sd_data == 0] = np.nan
    
    # Index data type is integer
    sd_TSS.index = sd_TSS.index.astype(int)
    sd_data['TSS'] = sd_TSS.loc[:,"TSS"]
    sd_data['LOI'] = sd_TSS.loc[:,"LOI"]
    return sd_data.round(2)







# Sampling data, arranged this way for loops purposses
sd = [
      
pd.concat([sd_time(2, 12), sd_data(5, 12, 3, 12)], axis = 1),
pd.concat([sd_time(20, 3), sd_data(24, 3, 16, 3)], axis = 1),
pd.concat([sd_time(29, 10), sd_data(34, 30, 20, 10)], axis = 1),
pd.concat([sd_time(45, 2), sd_data(71, 6, 31, 2)], axis = 1),
pd.concat([sd_time(52, 3), sd_data(84, 9, 34, 3)], axis = 1),
pd.concat([sd_time(61, 1), sd_data(100, 3, 38, 1)], axis = 1),
pd.concat([sd_time(68, 8), sd_data(110, 24, 40, 8)], axis = 1),
pd.concat([sd_time(82, 2), sd_data(141, 6, 49, 2)], axis = 1),
pd.concat([sd_time(90, 8), sd_data(154, 24, 52, 8)], axis = 1),
pd.concat([sd_time(104, 4), sd_data(185, 12, 61, 4)], axis = 1),
pd.concat([sd_time(114, 8), sd_data(204, 24, 66, 8)], axis = 1),
pd.concat([sd_time(128, 9), sd_data(235, 27, 75, 9)], axis = 1),
pd.concat([sd_time(143, 8), sd_data(269, 24, 85, 8)], axis = 1),
pd.concat([sd_time(157, 8), sd_data(300, 24, 94, 8)], axis = 1),
pd.concat([sd_time(171, 9), sd_data(331, 27, 103, 9)], axis = 1),
pd.concat([sd_time(186, 5), sd_data(365, 15, 113, 5)], axis = 1),
pd.concat([sd_time(197, 8), sd_data(387, 24, 119, 8)], axis = 1),
pd.concat([sd_time(211, 5), sd_data(418, 15, 128, 5)], axis = 1),
pd.concat([sd_time(222, 8), sd_data(439, 24, 134, 8)], axis = 1),
pd.concat([sd_time(236, 9), sd_data(469, 27, 143, 9)], axis = 1),
pd.concat([sd_time(251, 6), sd_data(502, 18, 153, 6)], axis = 1),
pd.concat([sd_time(263, 3), sd_data(526, 9, 160, 3)], axis = 1),
pd.concat([sd_time(272, 6), sd_data(541, 18, 164, 6)], axis = 1),
pd.concat([sd_time(284, 4), sd_data(565, 12, 171, 4)], axis = 1),
pd.concat([sd_time(294, 4), sd_data(583, 12, 176, 4)], axis = 1)
]






# Number of sampled events per month
months = []
for i in range(len(sd)):
    month = sd[i].Start[1]
    months.append(month)
# Convert the list of dates to a pandas Series
months = pd.DataFrame({'Timestamp': months})
# Group the dates by month
sampled_events_month = months.groupby(months['Timestamp'].dt.strftime('%Y-%m-01'))['Timestamp'].count()
sampled_events_month.index = pd.to_datetime(sampled_events_month.index)





# =============================================================================
# Supporting functions
# =============================================================================


def sd_list(column_name):
    '''
    Function to make a list of any parameter in the sd table
    
    Parameters
    ----------
    column_name : str
        Name of the column in sd df.

    Returns
    -------
    parameter : pandas.DataFrame
        df of all values of the parameter from sd.

    '''
    # Declaring arrays
    (start, end, parameter) = ([], [], [])

    
    for event in sd:
        for event_start, event_end, par in zip(event.Start, event.End, event[column_name]):
            start.append(event_start)
            end.append(event_end)
            parameter.append(par)
            
    # Making a table for scatter plots and analysis
    parameter_df = pd.DataFrame({"Start": start,
                         "End": end,
                         column_name: parameter})
    return parameter_df




def avg_cont_par_list(cont_data_set, column_name):
    '''
    Function to make a list of any continuous parameter for samples

    Parameters
    ----------
    cont_data_set : pandas.DataFrame
        Initial continuous dataset.
    column_name : str
        Name of the parameter in that df.

    Returns
    -------
    parameter : pandas.DataFrame
        df with continuous data averaged by times of taking samples.

    '''
    # Declaring arrays
    (S_P, E_P, rtp_mean_list) = ([], [], [])
    
    # Looping through sampledevents
    for event in sd:
        # Looping through samples within each event
        for sample_start, sample_end in zip(event.Start, event.End):
            rtp_mean = cont_data_set[sample_start:sample_end][column_name].mean()
            rtp_mean_list.append(rtp_mean)
            S_P.append(sample_start)
            E_P.append(sample_end)
    
    
    # Making a df
    parameter = pd.DataFrame({"Start": S_P,
                         "End": E_P,
                         column_name: rtp_mean_list})
    return parameter



def sampling_event(ev):
    '''
    Function for choosing sampling event automatically.
    Needed for simplifying referring to sample events,
    when making graphs

    Parameters
    ----------
    ev : str
        Just enter the number of the event.

    Returns
    -------
    s : str
        Index of the first sample in the list of all samples.
    e : str
        Index of the last sample in the list of all samples + 1.

    '''
    if ev == 1:
        st = 0
        end = 12
    elif ev == 2:
        st = 12
        end = 15
    elif ev == 3:
        st = 15
        end = 25
    elif ev == 4:
        st = 25
        end = 27
    elif ev == 5:
        st = 27
        end = 30
    elif ev == 6:
        st = 30 
        end = 31
    elif ev == 7:
        st = 31
        end = 39
    elif ev == 8:
        st = 39
        end = 41
    elif ev == 9:
        st = 41
        end = 49
    elif ev == 10:
        st = 49
        end = 53
    elif ev == 11:
        st = 53
        end = 61
    elif ev == 12:
        st = 61
        end = 70
    elif ev == 13:
        st = 70
        end = 78
    elif ev == 14:
        st = 78
        end = 86
    elif ev == 15:
        st = 86
        end = 95
    elif ev == 16:
        st = 95
        end = 100
    elif ev == 17:
        st = 100
        end = 108
    elif ev == 18:
        st = 108
        end = 113
    elif ev == 19:
        st = 113
        end = 121
    elif ev == 20:
        st = 121
        end = 130
    elif ev == 21:
        st = 130
        end = 136
    elif ev == 22:
        st = 136
        end = 139
    elif ev == 23:
        st = 139
        end = 145
    elif ev == 24:
        st = 145
        end = 149
    elif ev == 25:
        st = 149
        end = 153
    return st, end








