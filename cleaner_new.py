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
    df['SubModel'].fillna('NULL', inplace=True)
    df['OldSubModel'] = df['SubModel']

    ########################################################################

    print("\nLiters\n")
    df['EngineLiters'] = 0
    uniques = df['Model'].unique()
    regexLiters = "( \d[.]\dL| \d[.]\d| [\/]\d[.]\dL|-\d[.]\dL)"

    for line in uniques:
        print("\t", list(uniques).index(line), " of ", len(uniques), end="\r")

        if(re.findall(regexLiters, line)):
            res = re.sub(regexLiters, '', line)
            val = float(re.findall(r"(\d.\d)", re.findall(regexLiters, line)[0])[0])

            # print(line , " - " , re.findall(regexLiters, line), " - " , val)

            df['EngineLiters'] = np.where(df['Model'] == line, str(val), df['EngineLiters'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])


    # SUBMODEL CONTAINS INFO ABOUT LITERS AS WELL
    uniques = df['SubModel'].unique()
    regexLiters = "( \d[.]\dL| \d[.]\d| [\/]\d[.]\dL|-\d[.]\dL)"

    for line in uniques:
        print("\t", list(uniques).index(line), " of ", len(uniques), end="\r")

        if(re.findall(regexLiters, line)):
            res = re.sub(regexLiters, '', line)
            val = float(re.findall(r"(\d.\d)", re.findall(regexLiters, line)[0])[0])

            # print(line , " - " , re.findall(regexLiters, line), " - " , val)

            df['EngineLiters'] = np.where(df['SubModel'] == line, str(val), df['EngineLiters'])

    ########################################################################

    print("\nCylinders\n")
    df['NumCylinders'] = 0
    uniques = df['Model'].unique()

    regexV12 = "V12"
    regexV8 = "( V8| I8| 8C| V-8| I-8| V 8| I 8)"
    regexV6 = "( V6| I6| 6C| V-6| I-6| V 6| I 6)"
    regexV4 = "( V4| I4| 4C| V-4| I-4| V 4| I 4)"

    for line in uniques:
        print("\t", list(uniques).index(line), " of ", len(uniques), end="\r")

        if(re.findall(regexV8, line)):
            res = re.sub(regexV8, '', line)
            df['NumCylinders'] = np.where(df['Model'] == line, 8, df['NumCylinders'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

        if(re.findall(regexV6, line)):
            res = re.sub(regexV6, '', line)
            df['NumCylinders'] = np.where(df['Model'] == line, 6, df['NumCylinders'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])

        if(re.findall(regexV4, line)):
            res = re.sub(regexV4, '', line)
            df['NumCylinders'] = np.where(df['Model'] == line, 4, df['NumCylinders'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])
        
        if(re.findall(regexV12, line)):
            res = re.sub(regexV12, '', line)
            df['NumCylinders'] = np.where(df['Model'] == line, 12, df['NumCylinders'])
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
            res = re.sub(reg, '', line)
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
            df['WheelDrive'] = np.where(df['Model'] == line, "Front", df['WheelDrive'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])
            df.loc[df['Model'] == line] = res

        reg = " RWD"
        if(re.findall(reg, line)):
            res = re.sub(reg, '', line)
            df['4X4'] = np.where(df['Model'] == line, "NO", df['4X4'])
            df['WheelDrive'] = np.where(df['Model'] == line, "Rear", df['WheelDrive'])
            df['Model'] = np.where(df['Model'] == line, res, df['Model'])
            df.loc[df['Model'] == line] = res
    
    ##################################
    print("\nFixing Models\n")
    df['Model'] = np.where(df.Model.str.contains("1500.*SIERRA.*"), "1500 SIERRA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("1500.*SILVERADO.*"), "1500 SILVERADO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("2500.*SILVERADO.*"), "2500 SILVERADO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("300.*"), "300", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("3.2 CL.*"), "32 CL", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("3.2 TL.*"), "32 TL", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("350Z.*"), "350Z", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("626.*"), "626", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ACCORD.*"), "ACCORD", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ACCENT.*"), "ACCENT", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AERIO.*"), "AERIO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ALERO.*"), "ALERO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ALTIMA.*"), "ALTIMA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AMANTI.*"), "AMANTI", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ARMADA.*"), "ARMADA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AURA.*"), "AURA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AVALON.*"), "AVALON", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AVENGER.*"), "AVENGER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AVEO.*"), "AVEO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("AZERA.*"), "AZERA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("BEETLE.*"), "BEETLE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("BONNEVILLE.*"), "BONNEVILLE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CALIBER.*"), "CALIBER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CAMARO.*"), "CAMARO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CAMRY.*SOLARA.*"), "SOLARO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CAMRY.*"), "CAMRY", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CARAVAN SE.*"), "CARAVAN", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CANYON.*"), "CANYON", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CENTURY.*"), "CENTURY", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CHARGER.*"), "CHARGER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CIVIC.*"), "CIVIC", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("COMMANDER.*"), "COMMANDER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CENTURY.*"), "CENTURY", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("COBALT.*"), "COBALT", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CONCORDE.*"), "CONCORDE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("COOPER.*"), "COOPER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("COROLLA.*"), "COROLLA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("COUPE.*"), "COUPE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CR-V.*"), "CRV", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("CTS.*"), "CTS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("DURANGO.*"), "DURANGO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ECHO.*"), "ECHO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ECLIPSE.*"), "ECLIPSE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ELANTRA.*"), "ELANTRA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ENVOY.*"), "ENVOY", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ES300.*"), "ES300", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ESCAPE.*"), "ESCAPE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ESCORT.*"), "ESCORT", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("EXCURSION.*"), "EXCURSION", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("EXPEDITION.*"), "EXPEDITION", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("F150.*"), "F150", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("F250.*"), "F250", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("FOCUS.*"), "FOCUS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("FORENZA.*"), "FORENZA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("FREESTYLE.*"), "FREESTYLE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("FUSION.*"), "FUSION", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("G35.*"), "G35", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("G5.*"), "G5", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("G6.*"), "G6", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("GALANT.*"), "GALANT", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("GRAND AM.*"), "GRAND AM", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("GRAND CHEROKEE.*"), "GRAND CHEROKEE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("GRAND MARQUIS.*"), "GRAND MARQUIS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("GRAND PRIX.*"), "GRAND PRIX", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("GRAND VITARA.*"), "GRAND VITARA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("H3.*"), "H3", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("HHR.*"), "HHR", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("HIGHLANDER.*"), "HIGHLANDER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("I30.*"), "I30", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("I35.*"), "I35", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("IMPALA.*"), "IMPALA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("INTREPID.*"), "INTREPID", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ION.*"), "ION", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("JETTA.*"), "JETTA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("L SERIES.*"), "L SERIES", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("LACROSSE.*"), "LACROSSE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("LANCER.*"), "LANCER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("LE SABRE.*"), "LE SABRE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("^LS *"), "LS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("LS DO"), "LS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("LUCERNE.*"), "LUCERNE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("M35.*"), "M35", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("M45.*"), "M45", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MAGNUM.*"), "MAGNUM", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MALIBU.*"), "MALIBU", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MATRIX.*"), "MATRIX", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MAXIMA.*"), "MAXIMA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MAZDA3.*"), "MAZDA3", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MAZDA5.*"), "MAZDA5", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MAZDA6.*"), "MAZDA6", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MDX.*"), "MDX", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MIATA.*"), "MIATA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MILAN.*"), "MILAN", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MIRAGE.*"), "MIRAGE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MONTANA.*"), "MONTANA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MONTEGO.*"), "MONTEGO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MONTREY.*"), "MONTREY", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MPV.*"), "MPV", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MOUNTAINEER.*"), "MOUNTAINEER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MURANO.*"), "MURANO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("MUSTANG.*"), "MUSTANG", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("NAVIGATOR.*"), "NAVIGATOR", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("NEON.*"), "NEON", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ODYSSEY.*"), "ODYSSEY", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("OPTIMA.*"), "OPTIMA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("OUTLANDER.*"), "OUTLANDER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PACIFICA.*"), "PACIFICA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PATHFINDER.*"), "PATHFINDER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PILOT.*"), "PILOT", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PRIUS.*"), "PRIUS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PRIZM.*"), "PRIZM", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PROTEGE.*"), "PROTEGE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("PT CRUISER.*"), "PT CRUISER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("QUEST.*"), "QUEST", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("QX4.*"), "QX4", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RAIDER.*"), "RAIDER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RAV-4.*"), "RAV4", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("REGAL.*"), "REGAL", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RENO.*"), "RENO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RIO.*"), "RIO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RODEO.*"), "RODEO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RONDO.*"), "RONDO", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("RX300.*"), "RX300", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("S SERIES.*"), "S SERIES", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("S10.*"), "S10", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SABLE.*"), "SABLE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SANTA FE.*"), "SANTA FE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SEBRING.*"), "SEBRING", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SEDONA.*"), "SEDONA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SENTRA.*"), "SENTRA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SIENNA.*"), "SIENNA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SIERRA.*1500.*"), "1500 SIERRA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SILHOUETTE.*"), "SILHOUETTE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SONATA.*"), "SONATA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SPECTRA.*"), "SPECTRA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SPORTAGE.*"), "SPORTAGE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SRX.*"), "SRX", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("STRATUS.*"), "STRATUS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SUBURBAN 1500.*"), "SUBURBAN 1500", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SUBURBAN 2500.*"), "SUBURBAN 2500", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SUNFIRE.*"), "SUNFIRE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("SX4.*"), "SX4", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TAHOE.*"), "TAHOE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TAURUS.*"), "TAURUS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TC.*"), "TC", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TIBURON*"), "TIBURON", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TOWN CAR*"), "TOWN CAR", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TRAILBLAZER.*"), "TRAILBLAZER", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("TUCSON.*"), "TUCSON", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("VERONA.*"), "VERONA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("VIBE.*"), "VIBE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("VITARA.*"), "VITARA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("VUE.*"), "VUE", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("XA.*"), "XA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("XB.*"), "XB", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("XG 300.*"), "XG 300", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("XG 350.*"), "XG 350", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("XL-7.*"), "XL7", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("XTERRA.*"), "XTERRA", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("YARIS.*"), "YARIS", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("YUKON.*"), "YUKON", df['Model'])
    df['Model'] = np.where(df.Model.str.contains("ZEPHYR.*"), "ZEPHYR", df['Model'])

    
    df['NumDoors'] = 0
    df['NumDoors'] = np.where(df.SubModel.str.contains(".*5D.*"), 4, df['NumDoors'])
    df['NumDoors'] = np.where(df.SubModel.str.contains(".*4D.*"), 4, df['NumDoors'])
    df['NumDoors'] = np.where(df.SubModel.str.contains(".*3D.*"), 4, df['NumDoors'])
    df['NumDoors'] = np.where(df.SubModel.str.contains(".*2D.*"), 4, df['NumDoors'])
    
    #TODO 
    df['SubModel'] = np.where(df.SubModel.str.contains(".*SEDAN.*"), "SEDAN", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*SUV.*|.*CUV.*"), "SUV", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*JEEP.*"), "SUV", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*CAB.*"), "CAB", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*SR5.*"), "CAB", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*LARIAT.*"), "CAB", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*CONVERTIBLE.*"), "CONVERTIBLE", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*ROADSTER.*"), "CONVERTIBLE", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*SPYDER.*"), "CONVERTIBLE", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*HARDTOP.*|.*HARTOP.*"), "CONVERTIBLE", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*COUPE.*"), "COUPE", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*HAT.*BACK.*"), "HATCHBACK", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*MINIVAN.*"), "MINIVAN", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*WAGON.*"), "WAGON", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*CONVERTIBLE.*"), "CONVERTIBLE", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*CROSS.*"), "CROSSOVER", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*UTILITY.*"), "UTILITY", df['SubModel'])
    df['SubModel'] = np.where(df.SubModel.str.contains(".*SPORT.*"), "SPORT", df['SubModel'])
    
    regex="BAS|SEDAN|SUV|CAB|CONVERTIBLE|COUPE|HATCHBACK|MINIVAN|WAGON|UTILITY|SPORT"
    #IF NOT REGEX
    df['SubModel'] = np.where(~(df.SubModel.str.contains(regex)), "NULL", df['SubModel'])

    
    df['Nationality'] = np.where(df.Make.str.contains("ACURA"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("BUICK"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("CADILLAC"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("CHEVROLET"),"AMERICAN", df['Nationality']) 
    df['Nationality'] = np.where(df.Make.str.contains("CHRYSLER"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("DODGE"),"AMERICAN", df['Nationality']) 
    df['Nationality'] = np.where(df.Make.str.contains("FORD"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("GMC"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("HONDA"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("HYUNDAI"),"KOREAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("INFINITI"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("ISUZU"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("JEEP"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("KIA"),"KOREAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("LEXUS"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("LINCOLN"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("MAZDA"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("MERCURY"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("MINI"),"EUROPEAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("MITSUBISHI"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("NISSAN"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("OLDSMOBILE"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("PLYMOUTH"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("PONTIAC"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("SATURN"),"AMERICAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("SCION"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("SUBARU"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("SUZUKI"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("TOYOTA"),"JAPANESE", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("VOLKSWAGEN"),"EUROPEAN", df['Nationality'])
    df['Nationality'] = np.where(df.Make.str.contains("VOLVO"),"EUROPEAN", df['Nationality'])

    # NOTE 
    # is it real fundamental?
     
    # df['Make'] = np.where(df.Make.str.contains("ACURA"),"HONDA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("BUICK"),"GM", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("CADILLAC"),"GM", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("CHEVROLET"),"GM", df['Make']) 
    # df['Make'] = np.where(df.Make.str.contains("CHRYSLER"),"CHRYSLER", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("DODGE"),"CHRYSLER", df['Make']) 
    # df['Make'] = np.where(df.Make.str.contains("FORD"),"FORD", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("GMC"),"GMC", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("HONDA"),"HONDA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("HYUNDAI"),"HYUNDAI", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("INFINITI"),"NISSAN", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("ISUZU"),"ISUZU", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("JEEP"),"CHRYSLER", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("KIA"),"KIA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("LEXUS"),"TOYOTA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("LINCOLN"),"FORD", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("MAZDA"),"MAZDA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("MERCURY"),"FORD", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("MINI"),"BMW", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("MITSUBISHI"),"MITSUBISHI", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("NISSAN"),"NISSAN", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("OLDSMOBILE"),"GM", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("PLYMOUTH"),"CHRYSLER", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("PONTIAC"),"GM", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("SATURN"),"GM", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("SCION"),"TOYOTA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("SUBARU"),"SUBARU", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("SUZUKI"),"SUZUKI", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("TOYOTA"),"TOYOTA", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("VOLKSWAGEN"),"VOLKSWAGEN", df['Make'])
    # df['Make'] = np.where(df.Make.str.contains("VOLVO"),"VOLVO", df['Make'])



    # df['Nationality'] = np.where(df['Make'] == 'HYUNDAI', 'OTHER ASIAN', df['Nationality'])
    # df['Nationality'] = np.where(df['Make'] == 'TOYOTA', 'TOP LINE ASIAN', df['Nationality'])


    # REMOVING WEIRD OUTLIERS
    to_drop = df[df['VehBCost'] < 2].index.values
    df.drop(index=to_drop, inplace=True)
    train_ids.remove(to_drop)

    # REMOVING USELESS COLUMN
    df.drop(columns=['WheelTypeID'], inplace=True)
    df.drop(columns=['PRIMEUNIT'], inplace=True)
    df.drop(columns=['AUCGUART'], inplace=True)
    df.drop(columns=['OldModel'], inplace=True)
    df.drop(columns=['OldSubModel'], inplace=True)

    # NOTE
    # why not imputation?
    # 
    # FILL NAN WITH A DEFAULT CATEGORY
    df['Nationality'].fillna('AMERICAN', inplace=True)
    df['Size'].fillna('NULL', inplace=True)
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
