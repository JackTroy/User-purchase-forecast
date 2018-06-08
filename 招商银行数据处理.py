import pandas as pd
import numpy as np
import time

# 读取个人信息
train_agg = pd.read_csv('train_agg.csv', sep='\t')
train_flag = pd.read_csv('train_flg.csv', sep='\t')     # 用户flag 0-不买 1-买
train_log = pd.read_csv('train_log.csv', sep='\t')
type_data = train_log.drop_duplicates('USRID', 'first')     # 用户浏览类型

print('train flag shape:{},train agg shape:{}'.format(train_flag.shape, train_agg.shape))

# 分割点击
train_log['EVT_LBL_1'] = train_log['EVT_LBL'].apply(lambda x:x.split('-')[0])
train_log['EVT_LBL_2'] = train_log['EVT_LBL'].apply(lambda x:x.split('-')[1])
train_log['EVT_LBL_3'] = train_log['EVT_LBL'].apply(lambda x:x.split('-')[2])

train_log['OCC_TIM'] = train_log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
train_log = train_log.sort_values(['USRID','OCC_TIM'])
train_log['next_time'] = train_log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

train_log = train_log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})

train_data = pd.merge(train_agg, train_flag, on='USRID',how='left',copy=False)      # 合并
train_data = pd.merge(train_data,train_log,on=['USRID'],how='left',copy=False)
final = pd.merge(train_data, type_data[['USRID','TCH_TYP']], on='USRID',how='left',copy=False)
final.to_csv('final_data.csv',index=False)