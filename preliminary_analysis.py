import pandas as pd
import matplotlib.pyplot as plt 
from data_merging import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
import matplotlib.pyplot as plt
from data_split import * 
import seaborn as sns
from modelling import *
from evaluation import *
from sklearn.feature_selection import VarianceThreshold

def model_data(season_df):
    season_df.reset_index(drop = True, inplace = True)
    reg = season_df.REGION[0][:-1]
    train_df, test_df, target = kfold(season_df)
    train_df, test_df, num_cols_with_na = clean_data(train_df, test_df)
    
    if len(num_cols_with_na) > 0:
        train_df, test_df = knn_impute(train_df, test_df, num_cols_with_na)

    complete_train, complete_test = combine_data(season_df, train_df, test_df, target)
    norm_train, norm_test = normalise(complete_train, complete_test)
    evaluate(norm_train, norm_test, target, reg)
    plot_fig(norm_train, target, reg)
    return


def clean_data(train_df, test_df):
    #Remove columns that have too many constant values

    num_cols = train_df.select_dtypes('number').columns
    sel = VarianceThreshold(threshold=0)
    sel.fit(train_df[num_cols])  # fit finds the features with zero variance
    const = [x for x in train_df[num_cols].columns if x not in train_df[num_cols].columns[sel.get_support()]]
    remove_col = train_df[const].columns
    for elem in remove_col:
        train_df = train_df.drop(elem, axis=1)
        test_df = test_df.drop(elem, axis=1)

    num_cols = train_df.select_dtypes('number').columns
    num_cols_with_na = num_cols[train_df[num_cols].isna().mean() > 0]
    num_tecl_with_na = num_cols[test_df[num_cols].isna().mean() > 0]
    for index in num_tecl_with_na:
        if index not in num_cols_with_na:
            num_cols_with_na = num_cols_with_na.insert(0, index)
    return train_df, test_df, num_cols_with_na

def knn_impute(train_df, test_df, num_cols_with_na):
    # initialize imputer 
    imputer = KNNImputer(n_neighbors=3)

    # fit the imputer on train_df. pass only numeric columns.
    imputer.fit(train_df[num_cols_with_na])

    # transform the data using the fitted imputer
    train_df_knn_impute = imputer.transform(train_df[num_cols_with_na])
    test_df_knn_impute = imputer.transform(test_df[num_cols_with_na])

    # put the output into DataFrame. remember to pass columns used in fit/transform
    train_df_knn_impute = pd.DataFrame(train_df_knn_impute, columns=num_cols_with_na)
    test_df_knn_impute = pd.DataFrame(test_df_knn_impute, columns=num_cols_with_na)

    train_imputed_data = train_df_knn_impute.columns
    test_imputed_data = test_df_knn_impute.columns

    for col in train_imputed_data:
        train_df = train_df.drop(col, axis=1)
    
    train_df = train_df.reset_index(drop = True)
    train_df = pd.concat([train_df, train_df_knn_impute], axis=1)
    
    for col in test_imputed_data:
        test_df = test_df.drop(col, axis=1)
    test_df = test_df.reset_index(drop = True)
    test_df = pd.concat([test_df, test_df_knn_impute], axis=1)
    return train_df, test_df

def combine_data(season_df, train_df, test_df, target):
    target_df = pd.concat([season_df['Date'], target], axis=1)
    complete_train = train_df.join(target_df.set_index('Date'), on='Date')
    complete_test = test_df.join(target_df.set_index('Date'), on='Date')
    return complete_train, complete_test
    
def normalise(complete_train, complete_test):
    num_cols = complete_train.select_dtypes('number').columns
    train_df = complete_train[num_cols]
    test_df = complete_test[num_cols]
    
  
    # apply normalization techniques
    for column in train_df.columns:
        train_df[column] = (train_df[column] - train_df[column].mean()) / train_df[column].std()    
        test_df[column] = (test_df[column] - test_df[column].mean()) / test_df[column].std()                  

    return train_df, test_df
