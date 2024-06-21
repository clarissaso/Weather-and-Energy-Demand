import pandas as pd
import matplotlib.pyplot as plt 
from data_merging import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
import matplotlib.pyplot as plt
import seaborn as sns
from data_wrangling import *
from preliminary_analysis import *

def kfold(season_df):
    df = season_df.copy()
    target = df.iloc[:, -4:]
    print(df)
    y = target # Target variable   
    df.drop(target.columns, axis=1, inplace=True) # Removing target variable from training data
    X = df
    
    # Lets split the data into 5 folds.  
    # We will use this 'kf'(KFold splitting stratergy) object as input to cross_val_score() method
    kf =KFold(n_splits=10, shuffle=True, random_state=42)

    cnt = 1
    # split()  method generate indices to split data into training and test set.
    for train_index, test_index in kf.split(X, y):
        cnt += 1

    train_index_df = pd.DataFrame(index=train_index)
    train_df = pd.merge(df, train_index_df, left_index=True, right_index=True)
    test_index_df = pd.DataFrame(index=test_index)
    test_df = pd.merge(df, test_index_df, left_index=True, right_index=True)
    return train_df, test_df, target
