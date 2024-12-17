# eda_utils.py
# I was creating multiple correlation matrix heatmaps so I decided to just create a function to do it lol. I'll add more functions here as I need them.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr_matrix(df, nr_c, targ):
    """
    Plots a correlation matrix heatmap for the top 'nr_c' features most correlated with 'targ' in dataframe 'df'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        nr_c (int): Number of top correlated features to include.
        targ (str): The target column for correlation.
        
    Returns:
        None
    """
    # Compute the absolute correlation matrix
    corr_abs = df.corr().abs()
    
    # Select the top 'nr_c' features correlated with 'targ'
    top_features = corr_abs.nlargest(nr_c, targ)[targ].index
    
    # Compute the correlation matrix for the selected features
    cm = df[top_features].corr()
    
    # Plot the heatmap
    plt.figure(figsize=(nr_c / 1.5, nr_c / 1.5))
    sns.set(font_scale=1.25)
    sns.heatmap(
        cm,
        linewidths=1.5,
        annot=True,
        square=True,
        fmt='.2f',
        annot_kws={'size': 10},
        cmap="Blues"
    )
    plt.title(f'Correlation Matrix - Top {nr_c} Features Correlated with {targ}')
    plt.show()