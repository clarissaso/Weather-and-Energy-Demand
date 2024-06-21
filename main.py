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
from sklearn.feature_selection import VarianceThreshold
from evaluation import *

def main():

    price_demand_df = clean_price_demand_file()
    df = clean_weather_files()
    aut, win, spr, smr = merge_data(price_demand_df, df)

    aut = model_data(aut)

    
    return 0

#python -m main
main()
