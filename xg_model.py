import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import time
import os

def xgb_model(train_set_x, train_set_y, test_set_x, save=False):
    if 'USRID' in train_set_x.columns:
        train_set_x = train_set_x.drop(['USRID'], axis=1)
    if 'USRID' in test_set_x.columns:
        test_set_x = test_set_x.drop(['USRID'], axis=1)
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.02,
              'max_depth': 3,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 9,  # 2 3
              'silent':1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    if save:
        model.save_model(os.path.join('./saved_model',
            'xgboost_'+time.strftime("%H-%M-%S", time.localtime())+'.model'))
    return predict

def xgb_score(train, cv=5):
    data = train[0]
    target = train[1]
    if 'USRID' in target.columns:
        target.drop(['USRID'], axis=1, inplace=True)
    target = target.values
    target = target.ravel()
    auc_list = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=3)
    for train_index, test_index in skf.split(data, target):
        print('Train: %s | test: %s' % (train_index, test_index))
        X_train, X_test = data.loc[train_index], data.loc[test_index]
        y_train, y_test = target[train_index], target[test_index]

        pred_value = xgb_model(X_train, y_train, X_test)
        print(pred_value)
        print(y_test)

        pred_value = np.array(pred_value)
        pred_value = [ele + 1 for ele in pred_value]

        y_test = np.array(y_test)
        y_test = [ele + 1 for ele in y_test]

        fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)

        auc = metrics.auc(fpr, tpr)
        print('auc value:', auc)
        auc_list.append(auc)

    print('validate result:', np.mean(auc_list))

def save_result(train_data, test_data, result_name='test_result'):
    uid = test_data[0]['USRID'].reset_index()
    uid.drop(['index'], axis=1, inplace=True)
    if 'USRID' in train_data[1].columns:
        train_data[1].drop(['USRID'], axis=1, inplace=True)
    result = xgb_model(train_data[0], train_data[1], test_data[0])
    re = pd.concat([uid, pd.DataFrame(result)], axis=1)
    re.rename(columns={0:'RST'}, inplace=True)
    re.to_csv(os.path.join('./prediction', result_name + '.csv'), index=False, sep='\t')
    print('save succeed')
    return re
