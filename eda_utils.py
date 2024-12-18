# eda_utils.py
# I was creating multiple correlation matrix heatmaps so I decided to just create a function to do it hehehe. I'll add more functions here as I need them.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr_matrix(df, targ, nr_c=None):
    """
    Plots a correlation matrix heatmap. If nr_c is specified, shows top correlations with target.
    Otherwise shows correlations for all columns with target.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        targ (str): The target column for correlation.
        nr_c (int, optional): Number of top correlated features to include.
        
    Returns:
        None
    """
    
    corr_abs = df.corr().abs()
    
    # If nr_c is specified, select top correlated features
    if nr_c is not None:
        top_features = corr_abs.nlargest(nr_c, targ)[targ].index
        cm = df[top_features].corr()
        title = f'Correlation Matrix - Top {nr_c} Features Correlated with {targ}'
        figsize = (nr_c / 1.5, nr_c / 1.5)
        annot_size = 10
        cbar_kws = {'shrink': .5}
    # Else show all features  
    else:
        cm = corr_abs
        title = f'Correlation Matrix - All Features with {targ}'
        figsize = (20, 16)
        annot_size = 7
        cbar_kws = {'shrink': .95}  
    
    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.25)
    sns.heatmap(
        cm,
        linewidths = 0.5,
        annot = True,
        square = True,
        fmt = '.2f',
        annot_kws = {'size': annot_size},
        cmap = "Blues",
        cbar_kws = cbar_kws  
    )
    plt.title(title)
    plt.xticks(rotation = 45, ha = 'right')
    plt.yticks(rotation = 0)
    plt.tight_layout()
    plt.show()

def impute_missing_data(df):
    """
    Imputes missing data from dataset. 
    For categorical features where NA means absence of feature, fills with 'None'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    # Create a copy to avoid modifying original dataframe
    df = df.copy()
    
    # List of categorical features where NA means absence of feature
    na_means_none = [
        'Alley',          # No alley access
        'BsmtQual',       # No Basement
        'BsmtCond',       # No Basement
        'BsmtExposure',   # No Basement
        'BsmtFinType1',   # No Basement
        'BsmtFinType2',   # No Basement
        'FireplaceQu',    # No Fireplace
        'GarageType',     # No Garage
        'GarageFinish',   # No Garage
        'GarageQual',     # No Garage
        'GarageCond',     # No Garage
        'PoolQC',         # No Pool
        'Fence',          # No Fence
        'MiscFeature',    # No miscellaneous feature
        'MasVnrType'      # No masonry veneer
    ]
    
    # Fill NA with 'None' for these categorical features
    for col in na_means_none:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    
    
    return df