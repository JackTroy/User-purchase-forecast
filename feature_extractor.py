import numpy as np
import pandas as pd
import time

################## functions to obtain each features ##################
# notice 1:                                                           #
# functions args must be agg, log, flg                                #
# functions should return dataframe, no others type.                  #
# return feature must contain all USRID to assure final concatenation.#

# notice 2: for training set, all userid in log are also in agg.      #
# bur for testing set, not all userid in log are in agg as well.      #

CONTINUOUS = 1
DISCRETE = 2

def get_agg_features(agg, log, all_user_id):
    agg_features = pd.merge(all_user_id, agg, on=['USRID'], how='left')
    agg_features = agg_features.fillna(0)
    agg_types = [CONTINUOUS] * (agg.shape[1] - 1)
    for i in [2, 4, 5]:
        agg_types[i - 1] = DISCRETE 
    return agg_features, agg_types

def get_EVT_LBL_len(agg, log, all_user_id):
    EVT_LBL_len = log[['USRID', 'EVT_LBL']].groupby(by=['USRID'], as_index=False)['EVT_LBL']\
        .agg({'EVT_LBL_len': len})
    EVT_LBL_len = pd.merge(all_user_id, EVT_LBL_len, on=['USRID'], how='left')
    EVT_LBL_len = EVT_LBL_len.fillna(0)
    return EVT_LBL_len, [CONTINUOUS]

def get_EVT_LBL_set_len(agg, log, all_user_id):
    EVT_LBL_set_len = log[['USRID', 'EVT_LBL']].groupby(by=['USRID'], as_index=False)['EVT_LBL']\
        .agg({'EVT_LBL_set_len': lambda x: len(set(x))})
    EVT_LBL_set_len = pd.merge(all_user_id, EVT_LBL_set_len, on=['USRID'], how='left')
    EVT_LBL_set_len = EVT_LBL_set_len.fillna(0)
    return EVT_LBL_set_len, [CONTINUOUS]

def get_next_time_features(agg, log, all_user_id):
    log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    log = log.sort_values(['USRID','OCC_TIM'])
    log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
    log_time = log.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean':np.mean,
        'next_time_std':np.std,
        'next_time_min':np.min,
        'next_time_max':np.max
    })
    time_features = pd.merge(all_user_id, log_time, on=['USRID'], how='left')
    time_features.fillna(0)
    return time_features, [CONTINUOUS] * 4

def get_type(agg, log, all_user_id):
    type_data = log.drop_duplicates('USRID', 'first')[['USRID','TCH_TYP']]
    type_data = pd.merge(all_user_id, type_data, on=['USRID'], how='left')
    type_data = type_data.fillna(0)
    return type_data, [DISCRETE]

def get_click_count(agg, log, all_user_id):
    click_count = log['USRID'].value_counts()
    user_id = click_count.index
    user_click_count = click_count.values
    click_count = pd.DataFrame({
        'USRID': user_id,
        'user_click_times': user_click_count
    })
    click_count = pd.merge(all_user_id, click_count, on=['USRID'], how='left')
    click_count.fillna(0)
    return click_count, [CONTINUOUS]
############################ functions end ############################

FEATURE_LIST = [
    ('agg_features', get_agg_features),
    ('EVT_LBL_len', get_EVT_LBL_len),
    ('EVT_LBL_set_len', get_EVT_LBL_set_len),
    ('next_time_features', get_next_time_features),
    ('type_data', get_type),
    ('click_times', get_click_count)
]

def get_feature(agg, log, flg):
    features = flg.loc[:, 'USRID':'USRID']
    all_user_id = flg.loc[:, 'USRID':'USRID']
    feature_types = list()
    for feature in FEATURE_LIST:
        print(feature[0])
        feature_val, feature_type = feature[1](agg, log, all_user_id)
        features = pd.merge(features, feature_val, on=['USRID'], how='left')
        feature_types += feature_type
    return features, feature_type