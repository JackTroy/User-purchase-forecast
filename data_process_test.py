import pandas as pd
import numpy as np
import time
import os


def data_pro_test():

    if os.path.exists('pro_test.csv') and os.path.exists('pro_test_id.csv'):
        print('The data is exists .\nif data changed ,please delete the pro_test.csv .')
        test_data = pd.read_csv('pro_test.csv')
        usrid = pd.read_csv('pro_test_id.csv')
        return test_data, usrid

    test_agg = pd.read_csv('test_agg.csv', sep='\t')
    test_log = pd.read_csv('test_log.csv', sep='\t')
    logs = test_log
    type_data = test_log.drop_duplicates('USRID', 'first')     # 用户浏览类型

    print('test log shape:{},train agg shape:{}'.format(test_log.shape, test_agg.shape))

    EVT_LBL_len = test_log.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_len': len})
    EVT_LBL_set_len = test_log.groupby(by=['USRID'], as_index=False)['EVT_LBL']\
        .agg({'EVT_LBL_set_len': lambda x: len(set(x))})
    test_log['OCC_TIM'] = test_log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    test_log = test_log.sort_values(['USRID','OCC_TIM'])
    test_log['next_time'] = test_log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

    test_log = test_log.groupby(['USRID'],as_index=False)['next_time'].agg({
        'next_time_mean':np.mean,
        'next_time_std':np.std,
        'next_time_min':np.min,
        'next_time_max':np.max
    })
    test_log = pd.merge(test_log, EVT_LBL_len, on=['USRID'], how='left')
    test_log = pd.merge(test_log, EVT_LBL_set_len, on=['USRID'], how='left')
    test_data = pd.merge(test_agg,test_log,on=['USRID'],how='left',copy=False)
    final = pd.merge(test_data, type_data[['USRID','TCH_TYP']], on='USRID',how='left',copy=False)

    final['TCH_TYP'].fillna(-1,inplace=True)
    user_count = logs['USRID'].value_counts()
    count_array = pd.DataFrame(user_count)
    count_array['ss'] = count_array.index
    count_array['ss'] = count_array['USRID']
    count_array['USRID'] = count_array.index
    count_array.columns = ['USRID','times']
    final = pd.merge(final, count_array, on='USRID', how='outer')
    final.fillna(0,inplace=True)
    usrid = final[['USRID']]
    final.drop(['USRID'],axis=1,inplace=True)
    #final = pd.get_dummies(final, columns=['V2', 'V4', 'V5'])   #one-hot
    pd.DataFrame.to_csv(final, 'pro_test.csv',index=False)
    pd.DataFrame.to_csv(usrid, 'pro_test_id.csv', index=False)
    return final, usrid
