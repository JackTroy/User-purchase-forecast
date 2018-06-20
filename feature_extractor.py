import os
import time
import pickle
import numpy as np
import pandas as pd
from data_handler import get_data
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

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
    time_features = time_features.fillna(0)
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
    click_count = click_count.fillna(0)
    return click_count, [CONTINUOUS]

def get_average_EVENT_sequence_vector(agg, log, all_user_id):
    # no fucking time decay yet
    df_event_sequence = log['EVT_LBL'].apply(lambda x:x.split('-'))
    list_event_sequence = df_event_sequence.values.tolist()
    w2v_model = Word2Vec(list_event_sequence, size=5, window=1, min_count=1, negative=3,
                 sg=1, sample=0.001, hs=1, workers=4)
    df_event_seq_vec_sum = df_event_sequence.apply(
        lambda seq: np.sum([w2v_model.wv[mod] for mod in seq], axis=0))
    df_event_seq_vec_sum = pd.DataFrame(np.array(df_event_seq_vec_sum.values.tolist()))
    df_event_seq_vec_sum.columns = ['avg_vec_{}'.format(i) for i in range(5)]
    df_event_seq_vec_sum['USRID'] = log['USRID'].values
    df_usrid_average_seq_vec = df_event_seq_vec_sum.groupby(by=['USRID']).aggregate(np.mean).reset_index()
    df_usrid_average_seq_vec = pd.merge(all_user_id, df_usrid_average_seq_vec, on=['USRID'], how='left')
    df_usrid_average_seq_vec = df_usrid_average_seq_vec.fillna(0)
    return df_usrid_average_seq_vec, [CONTINUOUS] * 5

def get_dummy(total_events):
    '''one_hot_encoding batches'''
    pd.get_dummies(total_events.iloc[:1000000,:],columns=['0','1','2']).groupby(
        'USRID').apply(np.sum).to_csv('./data/fisrt_million.csv')
    pd.get_dummies(total_events.iloc[1000001:2000000,:],columns=['0','1','2']).groupby(
        'USRID').apply(np.sum).to_csv('./data/second_million.csv')
    pd.get_dummies(total_events.iloc[2000001:3000000,:],columns=['0','1','2']).groupby(
        'USRID').apply(np.sum).to_csv('./data/third_million.csv')
    pd.get_dummies(total_events.iloc[3000001:,:],columns=['0','1','2']).groupby(
        'USRID').apply(np.sum).to_csv('./data/final_million.csv')

def get_events_click_count(agg, log, all_user_id):
    regenerate = os.path.exists('./data/final_events.csv')
    if regenerate == False:
        events = log[['USRID','EVT_LBL']]
        #split the click-events to 3 columns
        events_spl = pd.DataFrame(events['EVT_LBL'].str.split('-').values.tolist())
        all_user_id = all_user_id.reset_index()
        total_events = pd.concat([all_user_id.iloc[:,1:],events_spl],axis=1)

        get_dummy(total_events)
        first = pd.read_csv('./data/fisrt_million.csv')
        second = pd.read_csv('./data/second_million.csv')
        third = pd.read_csv('./data/third_million.csv')
        final = pd.read_csv('./data/final_million.csv')
        # integrate the four data 
        agg = pd.concat([pd.concat([pd.concat([first,second],axis=0),third],axis=0),final],axis=0)
        agg.drop('USRID.1',axis=1,inplace=True)
        # groupby again avoid missing feature in the convergence
        event_count_data = agg.groupby('USRID').apply(np.sum)
        if 'USRID.1' in event_count_data.columns:
            event_count_data.drop('USRID.1', axis=1, inplace=True)
        event_count_data.fillna(-1)
        event_count_data.to_csv('./data/final_events.csv')
        return event_count_data
    else:
        event_count_data = pd.read_csv('./data/final_events.csv')
        if 'USRID.1' in event_count_data.columns:
            event_count_data.drop('USRID.1', axis=1, inplace=True)
        return event_count_data

def get_events_click_count_pca(agg, log, all_user_id):
    total_data = get_events_click_count(agg, log, all_user_id)
    if 'USRID.1' in total_data.columns:
        total_data.drop('USRID.1', axis=1, inplace=True)
    total_data = pd.merge(all_user_id, total_data, on='USRID', how='left')
    total_data.fillna(-1, inplace=True)
    total_data = total_data.drop('USRID', axis=1)
    
    pcas = PCA(n_components=60, random_state=0)
    event_data_pca = pcas.fit_transform(total_data)
    print("pca explained variance ratio sum:")
    print(sum(pcas.explained_variance_ratio_))
    event_data_pca = pd.DataFrame(event_data_pca)
    all_user_id = all_user_id.reset_index(drop=True)
    event_data_pca = pd.concat([all_user_id, event_data_pca], axis=1)
    return event_data_pca, [CONTINUOUS]

############################ functions end ############################

############################ Features #################################
FEATURE_LIST = [
    ('agg_features', get_agg_features),
    #('EVT_LBL_len', get_EVT_LBL_len), duplicate feature,
    ('EVT_LBL_set_len', get_EVT_LBL_set_len),
    ('next_time_features', get_next_time_features),
    ('type_data', get_type),
    ('click_times', get_click_count),
    # 06/13/18:34 added by jack_troy
    ('average_EVENT_sequence_vector', get_average_EVENT_sequence_vector),
    ('events_click_count_pca', get_events_click_count_pca)
]
########################## Features end ###############################

def get_features(regenerate=True):
    if regenerate:
        agg, log, flg = get_data()
        features = flg.loc[:, 'USRID':'USRID']
        all_user_id = flg.loc[:, 'USRID':'USRID']
        feature_types = list()
        for feature in FEATURE_LIST:
            print(feature[0])
            feature_val, feature_type = feature[1](agg, log, all_user_id)
            features = pd.merge(features, feature_val, on=['USRID'], how='left')
            feature_types += feature_type
        features.to_csv('./feature/features.csv')
        with open('./feature/feature_types', 'wb') as f:
            pickle.dump(feature_types, f)
        flg.to_csv('./feature/flg.csv')
    else:
        features = pd.read_csv('./feature/features.csv', index_col=0)
        with open('./feature/feature_types', 'rb') as f:
            feature_types = pickle.load(f)
        flg = pd.read_csv('./feature/flg.csv', index_col=0)

    features = features.reset_index(drop=True)
    flg = flg.reset_index(drop=True)
    train_features = features[flg['FLAG'] != -1]
    test_features = features[flg['FLAG'] == -1]
    train_flg = flg[flg['FLAG'] != -1]
    test_flg = flg[flg['FLAG'] == -1]

    train = [train_features, train_flg]
    test = [test_features, test_flg]
    return train, test, feature_types
    