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

def clean_file(file = "training"):

    if file == "training":
        df = pd.read_csv('training.csv')
    else:
        df = pd.read_csv('test.csv')

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

    df['CAAP'].fillna(df['AAAP'],inplace=True)
    df['CRAP'].fillna(df['ARAP'],inplace=True)
    df['CACP'].fillna(df['AACP'],inplace=True)
    df['CRCP'].fillna(df['ARCP'],inplace=True)

    PAC = ['WA' ,'OR','AK','HI','CA']
    MNT = ['MT' ,'WY','ID','NV','UT','CO','AZ','NM']
    WNC = ['ND' ,'MN','SD','IA','NE','KS','MO']
    WSC = ['OK' ,'AR','TX','LA']
    ENC = ['WI' ,'MI','OH','IN','IL']
    ESC = ['KY' ,'TN','MS','AL']
    MAT = ['NY' ,'PA','NJ']
    SAT = ['WV' ,'MD','DE','DC','VA','NC','SC','GA','FL']
    NEN = ['ME' ,'NH','VT','MA','CT','RI']

    sost = []
    for i in df['VNST']:
        if i in PAC:
            sost.append('PAC')
        elif i in MNT:
            sost.append('MNT')
        elif i in WNC:
            sost.append('WNC')
        elif i in WSC:
            sost.append('WSC')
        elif i in ENC:
            sost.append('ENC')
        elif i in ESC:
            sost.append('ESC')
        elif i in MAT:
            sost.append('MAT')
        elif i in SAT:
            sost.append('SAT')
        elif i in NEN:
            sost.append('NEN')
        else:
            sost.append('XXX')

    df['Region'] = sost

    sost = []

    for i in df['Trim']:
        if i == 'Bas':
            sost.append('YES')
        else:
            sost.append("NO")

    df['IsBase'] = sost

    df['WheelTypeID'] = df.groupby(['IsBase']).WheelTypeID.apply(lambda x: x.fillna(x.mode()[0]))
    df['TopThreeAmericanName'].fillna('OTHER', inplace=True)
    df['Nationality'].fillna('AMERICAN', inplace=True)
    df['Transmission'].fillna(df['Transmission'].mode()[0], inplace=True)
    df['Color'].fillna(df['Color'].mode()[0], inplace=True) #Sostituisco con la moda che è silver
    df['Size'] = df.groupby(['Make']).Size.apply(lambda x: x.fillna(x.mode()[0]))
    df['SubModel'] = df.groupby([df['Make'], df['Model']]).SubModel.apply(lambda x: x.fillna(x.mode()[0]))
    df['Trim'].fillna('Bas', inplace=True) # Andrea

    tresh = 1000
    if file == "training":
        to_delete = df[(df['AAAP'] < tresh) &
        (df['AACP'] < tresh) &
        (df['ARAP'] < tresh) &
        (df['ARCP'] < tresh) &
        (df['CAAP'] < tresh) &
        (df['CACP'] < tresh) &
        (df['CRAP'] < tresh) &
        (df['CRCP'] < tresh)].index.tolist()

        auction_delete = df[(df['AAAP'] < tresh) &
        (df['AACP'] < tresh) &
        (df['CAAP'] < tresh) &
        (df['CACP'] < tresh)].index.tolist()

        retail_delete = df[(df['CRAP'] < tresh) &
        (df['CRCP'] < tresh) &
        (df['ARAP'] < tresh) &
        (df['ARCP'] < tresh)].index.tolist()

        to_delete += auction_delete + retail_delete

        to_delete.append(40998)
        to_delete.append(53937)
        to_delete = list(dict.fromkeys(to_delete))
        new_df = df.drop(index=to_delete)
    else:
        new_df = df

    # new_df.drop(columns=['Trim', 'WheelType', 'PRIMEUNIT', 'AUCGUART', 'VehYear', 'VNZIP1', 'VNST'], inplace=True)
    new_df.drop(columns=['WheelType', 'PRIMEUNIT', 'AUCGUART', 'VehYear', 'VNZIP1'], inplace=True)
    
    if file == "training":
        new_df.dropna(inplace=True)
    else:
        new_df['CAAP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['CRAP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['CACP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['CRCP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['AAAP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['ARAP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['AACP'].fillna(new_df['VehBCost'],inplace=True)
        new_df['ARCP'].fillna(new_df['VehBCost'],inplace=True)

    sost_auct = []
    sost_ret = []
    prova_auc = []
    prova_ret = []
    prova_acquis = []
    prova_current = []
    trend =  []

    for i, row in new_df.iterrows():
        n_ret = 0
        n_auc = 0
        n_acquis = 0
        n_current = 0
        retail = 0
        current = 0
        acquis = 0
        auction = 0
        # auction
        if row['AAAP'] >= tresh:
            n_auc = n_auc + 1
            n_acquis = n_acquis + 1
            acquis = acquis + row['AAAP']
            auction = auction + row['AAAP']
        if row['AACP'] >= tresh:
            n_auc = n_auc + 1
            n_acquis = n_acquis + 1
            auction = auction + row['AACP']
            acquis = acquis + row['AACP']
        if row['CAAP'] >= tresh:
            n_auc = n_auc + 1
            n_current = n_current + 1
            current = current + row['CAAP']
            auction = auction + row['CAAP']
        if row['CACP'] >= tresh:
            n_auc = n_auc + 1
            n_current = n_current + 1
            current = current + row['CACP']
            auction = auction + row['CACP']
        #retail
        if row['ARAP'] >= tresh:
            n_ret = n_ret + 1
            n_acquis = n_acquis + 1
            acquis = acquis + row['ARAP']
            retail = retail + row['ARAP']
        if row['ARCP'] >= tresh:
            n_ret = n_ret + 1
            n_acquis = n_acquis + 1
            acquis = acquis + row['ARCP']
            retail = retail + row['ARCP']
        if row['CRAP'] >= tresh:
            n_ret = n_ret + 1
            n_current = n_current + 1
            current = current + row['CRAP']
            retail = retail + row['CRAP']
        if row['CRCP'] >= tresh:
            n_ret = n_ret + 1
            n_current = n_current + 1
            current = current + row['CRCP']
            retail = retail + row['CRCP']
            
        if file=="test" and (n_auc == 0 or n_ret == 0):
            sost_auct.append(row['VehBCost'])
            sost_ret.append(row['VehBCost'])
        else:
            sost_auct.append((auction / n_auc))
            sost_ret.append((retail / n_ret))
        
        if n_acquis == 0 or n_current == 0:
            trend.append(0)
        else:
            a = (current - acquis) / (current + acquis)
            trend.append(a)
        
        prova_auc.append(n_auc)
        prova_ret.append(n_ret)
        prova_acquis.append(n_acquis)
        prova_current.append(n_current)
            
    new_df['AuctionAVG'] = sost_auct
    new_df['RetailAVG'] = sost_ret
    new_df['Trend'] = trend

    #new_df.drop(columns=['AAAP', 'AACP', 'CAAP', 'CACP', 'ARAP', 'ARCP', 'CRAP', 'CRCP'], inplace=True)

    if file == "training":
        new_df.to_csv('training_cleaned.csv')
        print("written to training_cleaned.csv")
    else:
        new_df.to_csv('test_cleaned.csv')
        print("written to test_cleaned.csv")

def main():
    clean_file(file = sys.argv[1])

if __name__ == "__main__":
    main()



