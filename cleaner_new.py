# Import PlaidML before Keras!
import os
import plaidml.keras
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# 

import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys
import re
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
        'MMRAcquisitionAuctionAveragePrice': 'AAAP',
        'MMRAcquisitionRetailAveragePrice': 'ARAP',
        'MMRCurrentAuctionAveragePrice': 'CAAP',
        'MMRCurrentRetailAveragePrice': 'CRAP',
        'MMRAcquisitionAuctionCleanPrice': 'AACP',
        'MMRAcquisitonRetailCleanPrice': 'ARCP',
        'MMRCurrentAuctionCleanPrice': 'CACP',
        'MMRCurrentRetailCleanPrice': 'CRCP'
    }, inplace=True)

    df['PurchDate'] = pd.to_datetime(
        df['PurchDate'], infer_datetime_format=True)
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

    df['OldModel'] = df['Model']

    ########################################################################

    print("\nLiters\n")
    df['EngineLiters'] = 0
    uniques = df['Model'].unique()
    regexLiters = r"( \d[.]\dL)"

    for line in uniques:
        print("\t", list(uniques).index(line), " of ", len(uniques), end="\r")

        if(re.findall(regexLiters, line)):
            res = re.sub(regexLiters, '', line)
            val = float(re.findall(r"(\d.\d)", re.findall(regexLiters, line)[0])[0])
            df['EngineLiters'] = np.where(
                df['Model'] == line, str(val), df['EngineLiters'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

    ########################################################################

    print("\nCylinders\n")
    df['NumCylinders'] = 0
    uniques = df['Model'].unique()

    regexV8 = "( V8| I8| 8C| V-8| I-8| V 8| I 8)"
    regexV6 = "( V6| I6| 6C| V-6| I-6| V 6| I 6)"
    regexV4 = "( V4| I4| 4C| V-4| I-4| V 4| I 4)"

    for line in uniques:
        print("\t", list(uniques).index(line), " of ", len(uniques), end="\r")

        if(re.findall(regexV8, line)):
            res = re.sub(regexV8, '', line)
            df['NumCylinders'] = np.where(
                df['Model'] == line, 8, df['NumCylinders'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

        if(re.findall(regexV6, line)):
            res = re.sub(regexV6, '', line)
            df['NumCylinders'] = np.where(
                df['Model'] == line, 6, df['NumCylinders'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

        if(re.findall(regexV4, line)):
            res = re.sub(regexV4, '', line)
            df['NumCylinders'] = np.where(
                df['Model'] == line, 4, df['NumCylinders'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

    ########################################################################

    print("\nWheelDrive\n")
    df['4X4'] = 0
    df['WheelDrive'] = 0
    uniques = df['Model'].unique()
    for line in uniques:
        print("\t", list(uniques).index(line), " of ", len(uniques), end="\r")

        reg = "( AWD| 4WD)"
        if(re.findall(reg, line)):
            res = re.sub(regexV4, '', line)
            df['4X4'] = np.where(df['Model'] == line, "YES", df['4X4'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

        reg = " 2WD"
        if(re.findall(reg, line)):
            res = re.sub(reg, '', line)
            df['4X4'] = np.where(df['Model'] == line, "NO", df['4X4'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

        reg = " FWD"
        if(re.findall(reg, line)):
            res = re.sub(reg, '', line)
            df['4X4'] = np.where(df['Model'] == line, "NO", df['4X4'])
            df['WheelDrive'] = np.where(
                df['Model'] == line, "Front", df['WheelDrive'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])
            df.loc[df['Model'] == line] = res

        reg = " RWD"
        if(re.findall(reg, line)):
            res = re.sub(reg, '', line)
            df['4X4'] = np.where(df['Model'] == line, "NO", df['4X4'])
            df['WheelDrive'] = np.where(
                df['Model'] == line, "Rear", df['WheelDrive'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])
            df.loc[df['Model'] == line] = res


    df['Nationality'] = np.where(
        df['Make'] == 'HYUNDAI', 'OTHER ASIAN', df['Nationality'])
    df['Nationality'] = np.where(
        df['Make'] == 'TOYOTA', 'TOP LINE ASIAN', df['Nationality'])
    df['Nationality'].fillna('AMERICAN', inplace=True)
    df['Size'].fillna('NULL', inplace=True)


    # REMOVING WEIRD OUTLIERS
    to_drop = df[df['VehBCost'] < 2].index.values
    df.drop(index=to_drop, inplace=True)
    train_ids.remove(to_drop)

    # REMOVING USELESS COLUMN
    df.drop(columns=['WheelTypeID'], inplace=True)
    df.drop(columns=['PRIMEUNIT'], inplace=True)
    df.drop(columns=['AUCGUART'], inplace=True)
    df.drop(columns=['OldModel'], inplace=True)


    # FILL NAN WITH A DEFAULT CATEGORY
    df['Trim'].fillna('Bas', inplace=True)
    df['Color'].fillna('NOT AVAIL', inplace=True)
    df['Transmission'].fillna('AUTO', inplace=True)
    df['WheelType'].fillna('NULL', inplace=True)
    df['TopThreeAmericanName'].fillna(str(df['TopThreeAmericanName'].mode()[0]), inplace=True)
    


    ################################## ENCODING ##################################
    # TODO SUBMODEL
    df.drop(columns=['SubModel'], inplace=True)

    for col in list(df):
        coltype = str(df[col].dtype)
        if (coltype == "object"):
            df[col] = LabelEncoder().fit_transform(df[col])

    # Save Test Set
    test_cleaned = df[df.RefId.isin(test_ids)]
     # test_cleaned.to_csv('new_test_cleaned.csv')

    # On the Training Set we do the imputation
    train_cleaned = df[df.RefId.isin(train_ids)]

    ################################## DATA IMPUTATION ################################## 
    print(train_cleaned.isna().sum())

    columns = train_cleaned.columns
    
    imp = IterativeImputer(max_iter=100, random_state=random.randint(0,1000), sample_posterior=True, verbose=True)
    imp.fit(train_cleaned)
    res = imp.transform(train_cleaned)
    train_cleaned = pd.DataFrame(res , columns=columns)
    # train_cleaned.to_csv('new_train_cleaned.csv')


if __name__ == "__main__":
    main()
