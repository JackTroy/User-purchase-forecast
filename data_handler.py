import pandas as pd
import os

def get_data(path='./data'):
    train_agg = pd.read_csv(os.path.join(path, 'train_agg.csv'), sep='\t')
    test_agg = pd.read_csv(os.path.join(path, 'test_agg.csv'), sep='\t')
    agg = pd.concat([train_agg,test_agg], copy=False)

    train_log = pd.read_csv(os.path.join(path, 'train_log.csv'), sep='\t')
    test_log = pd.read_csv(os.path.join(path, 'test_log.csv'), sep='\t')
    log = pd.concat([train_log,test_log], copy=False)

    train_flg = pd.read_csv(os.path.join(path, 'train_flg.csv'), sep='\t')
    test_flg = pd.read_csv(os.path.join(path, 'submit_sample.csv'), sep='\t')
    test_flg['FLAG'] = -1
    test_flg = test_flg.drop(columns=['RST'])
    flg = pd.concat([train_flg,test_flg], copy=False)

    return agg, log, flg