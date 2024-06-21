import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preliminary_analysis import *

def plot_fig(norm_data, target, reg):
    features = norm_data.columns
    targets = target.select_dtypes('number').columns
    for target_i in targets:
        detect_and_remove_outliers(norm_data, features, target_i, reg, inline_delete= False)

def detect_and_remove_outliers(norm_data, features, target_i, reg, inline_delete= True):
    min_percentile= 0.001
    max_percentile= 0.999
    nrows= int(np.ceil(len(features)/2))
    ncols= 2 
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (24, nrows * 6))
    outliers = []
    cnt = 0
    for row in range (0, nrows):
        for col in range (0, ncols):
            # Outlier detection using percentile
            min_thresold, max_thresold = norm_data[features[cnt]].quantile([min_percentile, max_percentile])
            df_outliers = norm_data[(norm_data[features[cnt]] < min_thresold) | (norm_data[features[cnt]] > max_thresold)]

            # Updaing list of outliers
            outliers = outliers + df_outliers.index.tolist()

            # Plot feature vs target using scatter plot
            ax[row][col].scatter(x = norm_data[features[cnt]], y= norm_data[target_i])

            # Mark outlier records in same scatter plot
            ax[row][col].scatter(x= df_outliers[features[cnt]],  y=df_outliers[target_i], marker ="o", edgecolor ="red", s = 100)
            ax[row][col].set_xlabel(features[cnt])
            ax[row][col].set_ylabel(target_i)
            ax[row][col].set_title('Outlier detection for feature ' + features[cnt])

            if inline_delete: 
                norm_data = norm_data.drop(df_outliers.index.tolist())
                norm_data.reset_index(drop = True, inplace = True)

            cnt = cnt + 1
            if cnt >= len(features):
                break
    
    plt.show()
    plt.savefig('outlier_detection ' + target_i[:3] + ' on ' + reg + '.png')
    label = ' wth_outl'
    heatmap(norm_data, label, reg)

    unique_outliers= list(set(outliers))
    
    if inline_delete == False: 
        norm_data = norm_data.drop(unique_outliers)
        norm_data.reset_index(drop = True, inplace = True)
    
    
    # Lets visulaize the feature(after droping outliers) and target relationship using Regplot
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (24, nrows * 6))

    outliers = []
    cnt = 0
    for row in range (0, nrows):
        for col in range (0, ncols):
            sns.regplot(data=norm_data, x = features[cnt], y= target_i, scatter_kws={'alpha':0.2}, line_kws={'color': 'blue'}, ax = ax[row,col]) # scatter_kws and line_kws used to pass additional keyword argument to change transparancy and line color
            ax[row,col].set_title("Regplot after removing outlier's from feature " + features[cnt], fontsize = 12)
            cnt = cnt + 1
            if cnt >= len(features):
                break

    plt.show()
    plt.savefig('removed_outliers ' + target_i[:3] + ' on ' + reg + '.png')
    label = ' no_outl'
    heatmap(norm_data, label, reg)

    return 0
    
def heatmap(norm_data, label, reg):
    corr = norm_data.corr(method= 'pearson') # Compute pairwise correlation of columns, excluding NA/null values. pearson : standard correlation coefficient
    f, ax = plt.subplots(figsize=(25, 25))

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    ax = sns.heatmap(corr, vmin=-1, vmax=1, mask=mask, cmap=cmap, center=0, annot = True, square=True, linewidths=.5, cbar_kws= {"shrink": .5, 'orientation': 'vertical'})
    plt.show()

    plt.savefig('heatmap' + label + ' on ' + reg + '.png')
    
    return 
