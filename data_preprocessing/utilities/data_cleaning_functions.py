# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:09:02 2023

@author: nikraz
"""

import pandas as pd
import numpy as np






# Function to find gaps in the time series
def find_gaps(df, min_gap_duration = 2):
    '''
    
    Parameters
    ----------
    df : pandas.DataFrame
        df where gaps have to be identified.
    min_gap_duration : int, optional
        Minimal duration of the interval between datapoints
        to make it identifiable as a gap (minutes). 
        The default is 2.

    Returns
    -------
    gaps : pandas.FataFrame
        df of gaps' beginnings, ends and durations.

    '''
    gaps = []
    previous_date = None
    next_date = None
    
    # Define a threshold for a significant time gap 
    threshold = pd.Timedelta(minutes = min_gap_duration)
        
    for date in df.index:
        if previous_date is not None:
            time_gap = date - previous_date
            if time_gap > threshold:
                next_date = previous_date + time_gap
                gaps.append({'Start': previous_date, 'End': next_date, 'Duration': time_gap})
        previous_date = date
    gaps = pd.DataFrame(gaps)
    return gaps







# Function to delete slices of the dataset
def replace_rows_nan(df, l_ok, f_ok):
    '''
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset from which we drop rows.
    l_ok : str
        Last datapoint that is ok, after which the rows will be 
        replaced with NaN.
    f_ok : str
        First datapoint that is ok, until which rows will be 
        replaced with NaN.

    Returns
    -------
    data : pandas.DataFrame
        Original df with part of the rows replaced with a 
        single row filled with Nan.

    '''
    
    # To brake dataset in the graph when there were no records
    # Making empty row df
    empty_row = {f'Column{i+1}': [np.nan] for i in range(df.shape[1])}
    empty_row = pd.DataFrame(empty_row)
    # Setting column names the same as in original data
    empty_row.columns = df.columns   
    new_index = pd.date_range(start = pd.Timestamp(f_ok) - pd.Timedelta(seconds = 1), periods = len(empty_row))
    empty_row = empty_row.set_index(new_index)
    
    data = pd.concat([df[:l_ok], empty_row, df[f_ok:]])
    # print(data[l_ok: f_ok])
    return data




# Function for returning gaps table when all data are NaN
def find_nan_periods(df, measuring_interval):
    '''
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame where NaN periods have to be identified.
    measuring_interval: str
        Timedelta str specifying the timestep between measured datapoints in df

    Returns
    -------
    nan_periods : pandas.DataFrame
        DataFrame of NaN periods' beginnings, ends and durations.

    '''
    nan_periods = []
    nan_start = None
    nan_end = None
    
    for date, row in df.iterrows():
        if row.isnull().all():
            if nan_start is None:
                nan_start = date
        else:
            if nan_start is not None:
                nan_end = date - pd.Timedelta(measuring_interval)
                nan_periods.append({'Start': nan_start, 'End': nan_end, 'Duration': nan_end - nan_start + pd.Timedelta(measuring_interval)})
                nan_start = None
    if nan_start is not None:
        nan_end = df.index[-1]
        nan_periods.append({'Start': nan_start, 'End': nan_end, 'Duration': nan_end - nan_start + pd.Timedelta(measuring_interval)})

    nan_periods = pd.DataFrame(nan_periods)
    return nan_periods


def replace_values_with_nan(df, pre_timestep, suc_timestep):
    '''
    Replace non-NaN values in specified interval with NaN values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame where values need to be replaced.
    pre_timestep : pandas.Timestamp
        Timestep preceeding the interval.
    suc_timestep : pandas.Timestamp
        Timestep succeeding the interval.
    
    Returns
    -------
    modified_df : pandas.DataFrame
        DataFrame with non-NaN values replaced with NaN within the specified interval.
    '''
    modified_df = df.copy()
    mask = (modified_df.index > pre_timestep) & (modified_df.index < suc_timestep)
    modified_df[mask] = modified_df[mask].map(lambda x: np.nan if pd.notnull(x) else x)
    return modified_df











