# 1. Import library
import lib.step1ml_utilis_summaryPre_processing as pre
import lib.step2ml_utils_feature_selection as fea
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
##############################################################################################################
# A. Import thư viện
##############################################################################################################
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import matplotlib.pyplot as plt
import time
import requests
import json
st.set_option('deprecation.showPyplotGlobalUse', False)


#############################################################################################################
# Thư viện dự đoán giá
#############################################################################################################

# 1. Import library
import lib.step1ml_utilis_summaryPre_processing as pre
import lib.step2ml_utils_feature_selection as fea
import lib.step3ml_utils_model_design_testing_regression as reg
import lib.step4ml_utils_model_design_testing_explainability as exp
import lib.step5ml_utils_model_design_testing_visualize_models as vis
import lib.step6ml_utils_model_design_testing_geospatial_analysis as geo
import lib.step9ml_Text_Kmeans_Clustering as clu
import lib.step11ml_utilis_model_design_time_series as clu
import matplotlib
import matplotlib_inline


import warnings
warnings.filterwarnings("ignore")

#############################################################################################################
# Thư viện time series
#############################################################################################################
# 1. Import library
import lib.step1ml_utilis_summaryPre_processing as pre
import lib.step2ml_utils_feature_selection as fea
import lib.step3ml_utils_model_design_testing_regression as reg
import lib.step4ml_utils_model_design_testing_explainability as exp
import lib.step5ml_utils_model_design_testing_visualize_models as vis
import lib.step6ml_utils_model_design_testing_geospatial_analysis as geo
import lib.step9ml_Text_Kmeans_Clustering as clu
import lib.step11ml_utilis_model_design_time_series as ser
import matplotlib
import matplotlib_inline
import streamlit as st
import datetime
import lib.step14ml_streamlit_pages as pag
import pickle
#import hydralit as hy

import warnings
warnings.filterwarnings("ignore")

def load_data(dir_file):
    names = ['Id','Date','Total_Volume','Item_4046','Item_4225','Item_4770','Total_Bags','Small_Bags','Large_Bags','XLarge_Bags','type','year','region']
    df = pre.loadData(file_dir=dir_file, names=names)
    df = df.iloc[1: , :]
    data = df.copy()
    return data


def pro_precessing(data):
    
    # Chuyển kiểu dữ liệu cho thuộc tính
    # Chuyển đổi dữ liệu date
    lst_date = 'Date'
    #pre.changeToAstype(df=data, lst_float=lst_float, lst_int=lst_int)
    #pre.changeToAstype_date(df=data, feature_date=lst_date)

    fea_id = pre.change_feature_seriesToDataframe(df=data, lst_feature='Id', names='fea_id')

    # Diện tích
    fea_date = pre.change_feature_seriesToDataframe(df=data, lst_feature='Date', names='fea_date')
    fea_date = pre.date_add_feature(df=fea_date, feature_date='fea_date')

    # Total_Volume
    fea_total_Volume = pre.change_feature_seriesToDataframe(df=data, lst_feature='Total_Volume', names='fea_total_Volume')
    # Total_Volume
    fea_Item_4046 = pre.change_feature_seriesToDataframe(df=data, lst_feature='Item_4046', names='fea_item_4046')
    # Total_Volume
    fea_item_4225 = pre.change_feature_seriesToDataframe(df=data, lst_feature='Item_4225', names='fea_item_4225')
    # Total_Volume
    fea_item_4770 = pre.change_feature_seriesToDataframe(df=data, lst_feature='Item_4770', names='fea_item_4770')
    # Total_Volume
    fea_total_Bags = pre.change_feature_seriesToDataframe(df=data, lst_feature='Total_Bags', names='fea_total_Bags')

    # Total_Volume
    fea_small_bags = pre.change_feature_seriesToDataframe(df=data, lst_feature='Small_Bags', names='fea_small_bags')
    # Total_Volume
    fea_large_bags = pre.change_feature_seriesToDataframe(df=data, lst_feature='Large_Bags', names='fea_large_bags')
    # Total_Volume
    fea_xlarge_bags = pre.change_feature_seriesToDataframe(df=data, lst_feature='XLarge_Bags', names='fea_xlarge_bags')
    # Total_Volume
    fea_type = pre.change_feature_seriesToDataframe(df=data, lst_feature='type', names='fea_type')
    # Total_Volume
    fea_year = pre.change_feature_seriesToDataframe(df=data, lst_feature='year', names='fea_year')
    # Total_Volume
    fea_region = pre.change_feature_seriesToDataframe(df=data, lst_feature='region', names='fea_region')
    lst_concat = [data, fea_id, fea_date, fea_total_Volume, fea_Item_4046, fea_item_4225, fea_item_4770, fea_total_Bags, fea_small_bags, fea_large_bags, fea_xlarge_bags, fea_type, fea_year, fea_region]
    data_processing_final = pre.dataframe_concat(lst_concat=lst_concat)
    return data_processing_final


def pro_precessing2(df):

    lst_float = ['fea_total_Volume', 'fea_item_4046', 'fea_item_4225', 'fea_item_4770', 'fea_total_Bags', 'fea_small_bags', 'fea_large_bags', 'fea_xlarge_bags', 'Total_Volume'
        , 'Item_4046', 'Item_4225', 'Item_4770', 'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags',]
    lst_int = ['fea_id', 'fea_year', 'fea_month', 'fea_day', 'fea_weekofyear', 'fea_daily', 'Id', 'year', 'fea_year']
    lst_date1 = 'Date'
    lst_date2 = 'fea_date'

    pre.change_type_lst(df=df, lst_change=lst_float, choose='float')
    pre.change_type_lst(df=df, lst_change=lst_int, choose='int')
    pre.changeToAstype_date(df=df, feature_date=lst_date1)
    pre.changeToAstype_date(df=df, feature_date=lst_date2)

    # 1. Xóa dữ cột không cần thiết
    lst_drop = ['Id', 'Date', 'Total_Volume', 'Item_4046', 'Item_4225', 'Item_4770', 'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 
    'type', 'year', 'region', 'fea_year']
    
    data_pre_processing = df.drop(lst_drop, axis=1)

    # . Total_volumn_item = fea_item_4046 + fea_item_4225 + fea_item_4770
    data_pre_processing['total_volumn_item'] = data_pre_processing['fea_item_4046'] + data_pre_processing['fea_item_4225'] +data_pre_processing['fea_item_4770']
    # fea_total_Bags = fea_small_bags + fea_large_bags + fea_xlarge_bags
    data_pre_processing['fea_total_Bags_calculation'] = data_pre_processing['fea_small_bags'] + data_pre_processing['fea_large_bags'] +data_pre_processing['fea_xlarge_bags']
    data_pre_processing['check_fea_total_Bags'] = round(data_pre_processing['fea_total_Bags'],0) -  round(data_pre_processing['fea_total_Bags_calculation'],0)



    # Dự đoán kiểu dữ liệu của thuộc tính theo các cập bậc input/ output và continious/ categorical
    # Biến liên tục
    lst_input_number_continious = ['fea_total_Volume', 'fea_item_4046',
        'fea_item_4225', 'fea_item_4770', 'fea_total_Bags', 'fea_small_bags',
        'fea_large_bags', 'fea_xlarge_bags', 'total_volumn_item', 'fea_total_Bags_calculation',
        'check_fea_total_Bags', 'fea_month', 'fea_day', 'fea_weekofyear', 'fea_daily', 'fea_id' ]
    # Biến phân loại
    lst_input_numbers_categorical = []
    lst_input_object_categorical = ['fea_type', 'fea_region',]
    lst_input_object_text = []
    lst_input_date = ['fea_date']
    # Biến categorical 
    lst_output_categorical = []
    # Biến liên tục - continious
    lst_output_continious = []


    # Trường hợp 1: Lấy toán bồ dữ liệu
    lst_variable = lst_output_continious + lst_input_date + lst_input_object_categorical + lst_input_numbers_categorical + lst_input_number_continious +lst_input_object_text
    data_analysis_all = data_pre_processing[lst_variable]


    features_choose = [ 'fea_type', 'fea_region', 'fea_total_Volume', 'fea_item_4046', 'fea_item_4225', 'fea_item_4770', 'fea_total_Bags', 'fea_small_bags', 'fea_large_bags', 'fea_xlarge_bags', 'total_volumn_item', 'fea_month', 'fea_day', 'fea_weekofyear', 'fea_daily', 'fea_id']
    data_analysis_final = data_analysis_all[features_choose]
    data_analysis_final = pre.add_dummies(data_analysis_final, x="fea_type", dropx=True)
    data_analysis_final = pre.add_dummies(data_analysis_final, x="fea_region", dropx=True)

    lst_lientuc_chosen = ['fea_total_Volume', 'fea_item_4046', 'fea_item_4225', 'fea_item_4770', 'fea_total_Bags', 'fea_small_bags','fea_large_bags', 'fea_xlarge_bags', 'total_volumn_item']
    data_analysis_scaler = pre.robust_Scaler(df=data_analysis_final,lst_lientuc_chosen=lst_lientuc_chosen)
    data_analysis_scaler.index += 1 

    data_analysis_date = pre.change_feature_seriesToDataframe(df=df,lst_feature='Date', names='fea_date')
    lst = ['Id','Date','Total_Volume','Item_4046','Item_4225','Item_4770','Total_Bags','Small_Bags',
        'Large_Bags','XLarge_Bags','type','year','region']
    data_origin = df[lst]
    lst_concat = [data_analysis_date, data_analysis_final, data_analysis_scaler, data_origin]

    #import pandas as pd
    #data_model_final = pd.concat(lst_concat, ignore_index=True)
    data_model_final = pre.dataframe_concat(lst_concat=lst_concat)
    return data_model_final


def pre_processing_data(dir_file):
    df = load_data(dir_file=dir_file)
    data = pro_precessing(data=df)

    data_final = pro_precessing2(df=data)

    lst_k_best_option1 = ['fea_type_organic', 'fea_item_4046', 'fea_item_4225', 'fea_item_4770', 'fea_small_bags', 'fea_large_bags', 'fea_xlarge_bags', 'fea_month', 'fea_total_Volume', 'total_volumn_item', 'fea_total_Bags']

    final = data_final[lst_k_best_option1]

    return final


