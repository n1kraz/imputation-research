# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:10:26 2023

@author: n1kraz
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# Settings of the plots style
plt.style.use('seaborn-v0_8-whitegrid')

# Settings for the console - more convenient use with big tables
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def plot_par(df, par_column, start, end, par_max=16, dpi=300):
    '''
    Function for navigating through the dataset

    Parameters
    ----------
    df : pd.DataFrame
        df with flow data.
    par_column : str
        Column name within this df.
    start : str
        Specify beginning of the period to plot.
    end : str
        Specify end of the period to plot.
    par_max : int, optional
        Adjust in case if the flof is giant. The default is 16.

    Returns
    -------
    Flow plot.

    '''
    st = pd.Timestamp(start)
    en = pd.Timestamp(end)
    
    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.plot(df.index, df[par_column], label=f'{par_column}')
    
    # Formatting
    plt.xlabel('Time')
    plt.legend()
    plt.gca().set_xlim(st, en)
    plt.gca().set_ylim(0, par_max)
    plt.show()


    
def identify_wet_flow(df, flow_column, threshhold_1, threshhold_2, before_flow, after_flow):
    '''
    Function to identify biginning and end of the wet weather event

    Parameters
    ----------
    df : pd.DataFrame
        df with flow data.
    flow_column : str
        Column name within this df.
    threshhold_1 : float
        Approx. minimal flow at the beginning of the event.
    threshhold_2 : float
        Approx. minimal flow at the end of the event.
    before_flow : str
        Approx. time just before event starts.
    after_flow : str
        Approx. time just after event ends.

    Returns
    -------
    beginning : pd.Timestamp
        Beginning of the wet flow event.
    end : pd.Timestamp
        End of wet flow event.

    '''

    # Create an array of values
    flow = df[flow_column]
    # Roughly defyning wet flow
    period = flow[before_flow:after_flow]

    # Find the index where the value first reaches threshhold_1
    beginning = period.index[np.argmax(period >= threshhold_1)]
    
    # Reverse the series to find the end of the wet flow
    reversed_period = period.sort_index(ascending=False)
    end = reversed_period.index[np.argmax(reversed_period >= threshhold_2)]

    return beginning, end
    





def plot_wet_event(df, flow_column, event, dpi=300):
    '''
    Function for making a graph of the event

    Parameters
    ----------
    df : pd.DataFrame
        df with flow data.
    flow_column : str
        Column name within this df.
    event : tupple
        Two identified timestamps of beginning and end of the event.

    Returns
    -------
    Graph of the event with wet flow colored.

    '''
    start = event[0]
    end = event[1]
    
    all_rain = df[str(start-pd.Timedelta('2 hours')):str(end+pd.Timedelta('2 hours'))]
    flow_wet = df[start:end]
    
    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.plot(all_rain.index, all_rain[flow_column], label=f'All {flow_column}')
    plt.plot(flow_wet.index, flow_wet[flow_column], label=f'Wet {flow_column}')
    
    plt.xlabel('Time')
    plt.ylabel('Flow, m3/h')
    

    # Formating Datetime axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(plt.gca().get_xticklabels(), 
              rotation=30, 
              horizontalalignment='right')
    
    plt.legend()

