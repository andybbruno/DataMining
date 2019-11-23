import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys

from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from scipy.stats.stats import pearsonr

def main():
    df_train = pd.read_csv('training.csv')
    df_test = pd.read_csv('test.csv')
    train_ids = df_train['RefId'].tolist()
    test_ids = df_test['RefId'].tolist()
    df = pd.concat([df_train, df_test])
    assert len(df_train) + len(df_test) == len(df)

    df.rename(columns={ 
            'MMRAcquisitionAuctionAveragePrice':'AAAP',
            'MMRAcquisitionRetailAveragePrice':'ARAP',
            'MMRCurrentAuctionAveragePrice':'CAAP',
            'MMRCurrentRetailAveragePrice':'CRAP',
            'MMRAcquisitionAuctionCleanPrice': 'AACP',
            'MMRAcquisitonRetailCleanPrice': 'ARCP',
            'MMRCurrentAuctionCleanPrice': 'CACP',
            'MMRCurrentRetailCleanPrice': 'CRCP'
        },inplace=True)

    df['PurchDate'] = pd.to_datetime(df['PurchDate'], infer_datetime_format=True)
    year = []
    month = []
    day = []
    week_day = []
    for e in df['PurchDate']:
        year.append(e.year)
        month.append(e.month)
        day.append(e.day)
        week_day.append(e.day_name())
    df['PurchYear'] = year
    df['PurchMonth'] = month
    df['PurchDay'] = day
    df['PurchWeekDay'] = week_day
    
    df['Make'] = np.where(df['Make'] == 'TOYOTA SCION', 'SCION', df['Make'])
    df['Make'] = np.where(df['Make'] == 'HUMMER', 'GMC',  df['Make'])
    df['Make'] = np.where(df['Make'] == 'PLYMOUTH', 'DODGE',  df['Make'])

    df['Trim'].fillna('Bas', inplace=True)
    df['Color'].fillna('NOT AVAIL', inplace=True)
    df['Transmission'].fillna('AUTO', inplace=True)
    df['WheelType'].fillna('NULL', inplace=True)

    df['Nationality'] = np.where(df['Make'] == 'HYUNDAI', 'OTHER ASIAN', df['Nationality'])
    df['Nationality'] = np.where(df['Make'] == 'TOYOTA', 'TOP LINE ASIAN', df['Nationality'])
    df['Nationality'].fillna('AMERICAN', inplace=True)
    df['Size'].fillna('NULL', inplace=True)    

    df.drop(index = [40998], inplace=True)
    train_ids.remove(40998)

    df.drop(columns=['WheelTypeID'], inplace=True)

    test_cleaned = df[df.RefId.isin(test_ids)]
    train_cleaned = df[df.RefId.isin(train_ids)]

    #train_cleaned.to_csv('new_train_cleaned.csv')
    #test_cleaned.to_csv('new_test_cleaned.csv')
    

if __name__ == "__main__":
    main()

