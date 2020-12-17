import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

def unique_check_categorical(df, threshold):
    table = df.describe(include=['O'])
    highly_unique = []
    for col in table.columns:
        f = table.loc['freq',col]/table.loc['count',col]
        if f > threshold:
            highly_unique.append(col)
            print(col, ', percentage of top unique value = %.2f %%' %(100*f))
    if len(highly_unique) == 0:
        print('There is no column has too high percentage of a value')


def unique_check_numerical(df, threshold):
    variables_detected = []
    for i in df:
        if (df[i].value_counts()[0] / len(df)) > threshold:
            variables_detected.append(i)
    return variables_detected


#find features with correlations greater than 0.9 in order to remove them
def correlation_check(df, limit = 0.90, drop = False):
    corr = df.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_no_diag = corr.where(mask)
    coll = [c for c in corr_no_diag.columns if any(abs(corr_no_diag[c]) > limit)]        
    #df.drop(coll,axis = 1,inplace=True)
    print('High correlation columns are:', coll)



# checking the value counts
def check_value_counts(df):
    for i in df.columns:
        print(i)
        print(df[i].value_counts()[:5])
        print('-------------------')

def boxplot_func(df):
    plot_features = df.columns
    #Prepare figure layout
    rows = int(math.ceil(df.shape[1]/8))
    sns.set()
    fig, axes = plt.subplots(nrows = rows, ncols=8, figsize=(16,4*rows))
  
    # Draw the boxplots
    for i in zip(axes.flatten(), plot_features):
        sns.boxplot(data=df.loc[:,i[1]], ax=i[0])      
        i[0].set_title(i[1])
        i[0].set_ylabel("")
        for tick in i[0].get_xticklabels():
            tick.set_rotation(-25)
    
    # Finalize the plot
    plt.subplots_adjust(wspace=2,hspace = 0.5)
    fig.suptitle("Box plots", fontsize=25)
    sns.despine(bottom=True)
    plt.show()
    
def histplot_func(df):
    plot_features = df.columns
    #Prepare figure layout
    rows = int(math.ceil(df.shape[1]/8))
    sns.set()
    fig, axes = plt.subplots(nrows = rows, ncols=8, figsize=(24,2*rows))
       
    # Draw the boxplots
    for i in zip(axes.flatten(), plot_features):
        sns.histplot(data=df.loc[:,i[1]], bins=20,ax=i[0])
        i[0].set_xlabel(i[1])
        i[0].set_ylabel("")
      
    # Finalize the plot
    plt.subplots_adjust(wspace=0.5,hspace = 0.5)
    fig.suptitle("Histograms", fontsize=15)
    sns.despine(bottom=True)
    plt.show()
    
def corrheatmap(dataframe):
    sns.set(style="white")

    # Compute the correlation matrix
    corr = dataframe.corr() #Getting correlation of numerical variables

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool) #Return an array of zeros (Falses) with the same shape and type as a given array
    mask[np.triu_indices_from(mask)] = True #The upper-triangle array is now composed by True values

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True) #Make a diverging palette between two HUSL colors. Return a matplotlib colormap object.

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, annot=True,annot_kws={"fontsize":10}, linewidths=.5,vmin = -1, vmax = 1, ax=ax)

    # Layout
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Correlation matrix", fontsize=15)
    plt.yticks(rotation=0)
    # Fixing the bug of partially cut-off bottom and top cells
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values

    plt.show()

# function to update selected features dataframe everytime we adjust the data
def update_selected_feats_list(df, features_list):
    selected_feats = features_list.loc[features_list['FeaturesName'].isin(df.columns),:]
    return selected_feats

def IQR_OutlierRemover(dfn):
    #give the function the column that you want to filter and the IQR multiplicator    
    # Compute the IQR
    q1= df.quantile(0.25)
    q3= df.quantile(0.75)
    Iqr = q3 - q1

    # Compute upper and lower limit (lower_limit = Q1 -1.5*IQR | upper_limit = Q3 + 1.5*IQR)
    lower_lim = q1 - n*Iqr
    upper_lim = q3 + n*Iqr

    initial_len = df.shape[0]

    df = df[~((df[column] < lower_lim) | (df[column] > upper_lim))]

    len_afterremov = df.shape[0]

    print('Percentage of data kept after removing outliers:', np.round(len_afterremov / initial_len, 4))

    return df