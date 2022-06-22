import numpy as np
import dask.dataframe as dd
import pandas as pd
from datetime import date
import datetime


def prepare_func(data_path, features_path):

    data_df = pd.read_csv(data_path)
    features = pd.read_csv(features_path, sep='\t')
    
    data_df = data_df.drop(['Unnamed: 0'],axis=1)
    features = features.drop(['Unnamed: 0'],axis=1)
    
    data_df['date'] =  data_df['buy_time'].apply(lambda x: date.fromtimestamp(x))
    features['date'] =  features['buy_time'].apply(lambda x: date.fromtimestamp(x))

    data_df = data_df.sort_values(by="buy_time")
    features = features.sort_values(by="buy_time")

    data = data_df.merge(features, on=['id'], how = 'left')

    data['not_correct'] = data.apply(lambda x: 1 if x['buy_time_y'] >= x['buy_time_x'] else 0, axis=1)
    col_n_cor = list(data.iloc[:,5:].columns)
    data.loc[data['not_correct'] == 1, col_n_cor] = 0

    data = data.sort_values(by="buy_time_y", ascending=False)
    data = data.drop_duplicates(subset = ['id', 'vas_id'],  keep = 'first')
    
    data['month'] = data['date_x'].apply(lambda x: pd.to_datetime(x).month)
    data['number_of_week'] = data['date_x'].apply(lambda x: pd.to_datetime(x).day // 7)
    
    data.drop(columns=['date_y', 'buy_time_y', 'date_x', 'not_correct'], axis=1, inplace=True)
    
    return data