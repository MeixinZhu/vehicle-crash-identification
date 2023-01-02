#!/usr/bin/env python
# coding: utf-8

import pdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
warnings.filterwarnings('ignore')
import generate_train_data
from generate_train_data import prepare

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
fc_parameters = {
"c3": [{"lag": lag} for lag in [2]],
    
"augmented_dickey_fuller": [{"attr": "teststat"}],

"energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 4}],

"fourier_entropy":  [{"bins": x} for x in [ 20]],
"permutation_entropy":  [{"tau": 1, "dimension": x} for x in [3, 5, 7]]}


def predict_a_des():
    print('Generate A round test predict based on descriptive features')
    temp = np.load('../user_data/train_des.npz')
    train_x, train_y = temp['x'], temp['y']

    X, y = np.nan_to_num(train_x, posinf = 10, neginf = -10), np.array(train_y)
    from sklearn.utils import shuffle
    X, y = shuffle(X, y)    

    model = AdaBoostClassifier(n_estimators = 20)
    model.fit(X, y)
    y_ = model.predict(X)
    score = f1_score(y, y_)
    print(score)

    window = 40

    acc_list = []
    for vehno in tqdm(range(121, 211)):
        acc_flag = False
        acc_time = ''
        
        data = prepare(vehno, 'test_allv2')
        battery_change = data[data['battery_pos_on'].diff() == -1]
        
        max_prob = -1
        
        for idx in battery_change.index:
            sample = data[idx-window: idx+window].reset_index(drop = True)

            # create features
            neg_con = int(sample['battery_neg_on'].iloc[0] == 1 and sample['battery_neg_on'].iloc[-1] == 0)
            occ_con = int(sample['driver_occ_on'].iloc[0] == 1 and sample['driver_occ_on'].iloc[-1] == 0)
            spd_con = int(np.median(sample['speed'][-10:]) == 0)
            current_con = int(np.median(sample['current'][-10:]) == 0)
            low_vol_drop = int(np.min(data['low_voltage'][idx-3:idx+3].diff()[1:]) < -0.9)

            des = sample.describe() 
            col_list = ['accelerator', 'torque', 'low_voltage', 'current', 'voltage', 'speed', 'accel', 'battery_pos_on',
                    'battery_neg_on', 
                'leaving_warn', 'seatbelt_on',
                'hand_brake_on',
                'gear_back', 'gear_parking', 'jerk', 'accelerator_diff',
        'low_voltage_diff', 'occ_diff', 'key_diff', 'gear_diff', 'brake_diff']

            des_fea = des[col_list][1:].values.reshape(-1)

            sample = sample[col_list]
            sample['id'] = 1
            ts_fea = extract_features(sample.replace([np.inf, -np.inf],0).fillna(0), column_id = 'id',
                                    default_fc_parameters=fc_parameters, n_jobs = 0, disable_progressbar=True)

            feature = np.append(np.array([neg_con, occ_con, spd_con, current_con, low_vol_drop]), des_fea)
            feature = np.append(feature, ts_fea.values)

            x_data = np.nan_to_num([feature], posinf = 100, neginf = -100)

            pre = model.predict(x_data)[0]
            prob = model.predict_proba(x_data)[0][1]
        
            if max_prob < prob:
                max_prob = prob
                
                acc_flag = True
                acc_time = data['time'][idx]
                
        acc_list.append([vehno, int(acc_flag), str(acc_time), max_prob])
        # print([vehno, int(acc_flag), str(acc_time), max_prob])

    model_res = pd.DataFrame(acc_list, columns = ['vehno','Label','CollectTime', 'prob'])
    out = model_res.rename(columns = {'vehno': '车号'})
    out.to_csv('../user_data/model_des_a_predict.csv', index = False)