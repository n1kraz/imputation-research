# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:06:12 2023

@author: n1kraz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
import matplotlib.dates as mdates





''' Visualizing continuous dataset per month for the duration of the campaign'''


# =============================================================================
# # Before correction
# =============================================================================


def plot_m_raw(raw_data_df, days_per_month):
    '''
    

    Parameters
    ----------
    raw_data_df : pd.DataFrame
        df of data BEFORE correction and split into wet and dry flows.

    Returns
    -------
    Bar graph of amount of data measured per month in units of time (days).

    '''


    # Indeces
    days_per_month.index = days_per_month.index.to_timestamp()
    
    plt.figure(figsize=(10, 6))
    plt.bar(days_per_month.index, days_per_month.values, width=20)
    plt.xlabel('\nDate')
    plt.ylabel('Amount of data, days')
    plt.title('Amount of data per month for the duration of the campaign\n', fontsize = 14)
    # Set X ticks to appear for every month
    plt.gca().xaxis.set_major_locator(MonthLocator())
    # Rotating labels
    plt.setp(plt.gca().get_xticklabels(), 
                  rotation = 90, 
                  horizontalalignment='center')
    
    plt.tight_layout()



# =============================================================================
# # After correction
# =============================================================================


def m_month(f_w, f_d, sampled_events):
    '''
    

    Parameters
    ----------
    f_w : pd.DataFrame
        df with the wet flow.
    f_d : pd.DataFrame
        df with the dry flow.

    Returns
    -------
    merged_df : pd,DataFrame
        df of amount of data per month measured in time units (days).
    m_wet_tot : int
        Total amount of data corresponding to the wet flow periods (days).
    m_dry_tot : int
        Total amount of data corresponding to the dry periods (days).

    '''
    # Groupping by month
    m_wet_month = f_w.groupby(f_w.index.to_period('M')).size()
    m_wet_month.index = m_wet_month.index.to_timestamp()
    # Transforming number of records into time (days)
    m_wet_month = m_wet_month/2880
    
    m_dry_month = f_d.groupby(f_d.index.to_period('M')).size()
    m_dry_month.index = m_dry_month.index.to_timestamp()
    # Transforming number of records into time (days)
    m_dry_month = m_dry_month/2880
    
    # Number of sampled events per month

    
    # Merge the grouped data for plotting
    m_wet_dry_month = pd.concat([m_wet_month, m_dry_month, sampled_events], axis=1)
    m_wet_dry_month.rename(columns={0: 'wet_flow',
                                    1: 'dry_flow',
                                    'Timestamp': 'num_sampled_events'}, 
                                    inplace=True)
    
    # Adding missing months to the df
    # Create a complete date range spanning the entire period
    complete_date_range = pd.date_range(start = m_wet_dry_month.index.min(), end = m_wet_dry_month.index.max(), freq='MS')
    # Create a new DataFrame with the complete date range
    complete_df = pd.DataFrame(index = complete_date_range)
    # Merge the original DataFrame with the complete DataFrame using 'left' join
    merged_df = complete_df.merge(m_wet_dry_month, how='left', left_index=True, right_index=True)
    
    # Total number of days measured during the campaign duration
    m_wet_tot = sum(m_wet_month)
    m_dry_tot = sum(m_dry_month)
    
    return merged_df, m_wet_tot, m_dry_tot
    
    


def plot_m_corr(f_w, f_d, sampled_events):
    '''
    

    Parameters
    ----------
    f_w : pd.DataFrame
        df with the wet flow.
    f_d : pd.DataFrame
        df with the dry flow.

    Returns
    -------
    Returns staked bar plot of amount of data after dataset correction.

    '''


    m_wet_dry_month = m_month(f_w, f_d, sampled_events)[0]
    m_wet_tot = m_month(f_w, f_d, sampled_events)[1]
    m_dry_tot = m_month(f_w, f_d, sampled_events)[2]

    # Making plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # X axis
    x_labels = m_wet_dry_month.index.strftime('%Y-%m')


    # Create a stacked bar plot of amount of data
    ax.bar(x_labels, m_wet_dry_month['dry_flow'], color = 'orange', alpha = 0.7)
    ax.bar(x_labels, m_wet_dry_month['wet_flow'], color = 'b', alpha = 0.7)


    # Create a secondary y-axis (inverted x-axis) for the second bar graph
    ax_top = ax.twinx()
    # Create the bar plot of number of events sampled hanging from the top
    ax_top.bar(x_labels, 
               m_wet_dry_month['num_sampled_events'], 
               color = 'r', 
               alpha = 0.7, 
               label = 'Number of sampled wet flow events per month')



    # Title
    plt.title('Amount of data per month for the duration of the campaign\n', fontsize = 14)

    # Set labels and title
    ax.set_xlabel('\nDate')
    ax.set_ylabel('Amount of data, days')
    ax_top.set_ylabel('Number of sampled events')

    # Legend 
    ax.legend([f'Dry periods. Total {int(m_dry_tot)} days {int((m_dry_tot - int(m_dry_tot))*24)} h',
               f'Wet flow. Total {int(m_wet_tot)} days {int((m_wet_tot - int(m_wet_tot))*24)} h'],
              loc = "lower left", bbox_to_anchor = (0, -0.3))
    ax_top.legend(loc = "lower left", bbox_to_anchor = (0, -0.34))


    # Formatting axes
    # Making some space for the second bar graph
    ax.set_ylim(0, 40)
    ax_top.set_ylim(0, 16)
    
    # Switching off the second grid
    ax_top.grid(False)
    
    # Rotating labels
    plt.setp(ax.get_xticklabels(), 
                  rotation=90, 
                  horizontalalignment='center')

    # Set the x-axis labels
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    
    # Invert the y-axis to make bars on top hang from the top
    ax_top.invert_yaxis()

    # Tight layout
    fig.tight_layout()

    # Show the plot
    plt.show()



# =============================================================================
# # All flow data
# =============================================================================

def plot_wet_dry_flow(f_w, f_d):
    '''
    

    Parameters
    ----------
    f_w : pd.DataFrame
        df with the wet flow.
    f_d : pd.DataFrame
        df with the dry flow.

    Returns
    -------
    Graph of the flow data for the whole duration of the campaign.

    '''
    plt.figure(figsize=(10, 6))
    plt.plot(f_d.index, f_d['Flow_m3h_Avg'], label='Dry periods')
    plt.plot(f_w.index, f_w['Flow_m3h_Avg'], label='Wet Flow')
    plt.xlabel('Time')
    plt.ylabel('Flow, m3/h')
    plt.legend()
    plt.show()










# =============================================================================
# # plotting any parameter for any timeperiod
# =============================================================================

def plot_event_par(df, par_column, st, end, margin = '0 hours'):
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        df from which the parameter will be taken.
    par_column : str
        Name of the parameter column in that df.
    st : str
        Beginning of the period to be plotted.
    end : str
        End of the period to be plotted.
    margin : str, optional
        In case if period is strictly defined through accessing it through 
        function, this allows to add time margin ahead and after the period. 
        The default is '0 hours'.

    Returns
    -------
    Plot of the parameter change for the time interval.

    '''
    # Adding time margin
    st = pd.Timestamp(st) - pd.Timedelta(margin)
    end = pd.Timestamp(end) + pd.Timedelta(margin)
    
    
    # Plot
    plt.plot(df[par_column][str(st):str(end)].index, df[par_column][str(st):str(end)].values)
    
    # Changing x axis format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Formating Datetime axis
    plt.setp(plt.gca().get_xticklabels(), 
              rotation=30, 
              horizontalalignment='right')
    plt.ylabel(par_column)
    plt.xlabel('\nTime')
    
    plt.tight_layout()
    
    plt.show()





# Graph of frequency of datapoints occuring in the range of parameter
def graph_par_range(df, par1 = 'Turb_FNU_Avg', par2 = 'DO_temp_Avg', par3 = 'Conduc_conduc_Avg', par4 = 'DO_mg_L_Avg', par5 = 'pH_pH_Avg', par6 = 'Flow_m3h_Avg'):
    fig2, ax2 = plt.subplots(2, 3, figsize=(8,  5), dpi=300)
    ax2 = ax2.flatten()
    for i, c in enumerate([par1,
                          par2,
                          par3,
                          par4,
                          par5,
                          par6]):
        df.hist(column = c,
                ax = ax2[i],
                bins = 68,
                color = 'c',
                edgecolor = 'w',
                alpha = 0.65,
                log = True)
        ax2[i].axvline(df[c].mean(),
                       color = 'k',
                       linestyle = 'dashed',
                       linewidth = 1)
        ax2[i].set_ylabel('Frequency')
        ax2[i].set_xlabel(c)
        ax2[i].set_title(None)
    fig2.suptitle('Distribution of values of the parameters', fontsize = 18)




