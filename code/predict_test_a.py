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

def predict_a():
    print('Generate A test predict')
    temp = np.load('../user_data/train_time.npz')
    train_x, train_y = temp['x'], temp['y']

    X, y = np.nan_to_num(train_x, posinf = 10, neginf = -10), np.array(train_y)
    from sklearn.utils import shuffle
    X, y = shuffle(X, y)    

    model = AdaBoostClassifier(n_estimators = 20)
    model.fit(X, y)
    y_ = model.predict(X)
    score = f1_score(y, y_)
    print(score)

    window = 10
    acc_list = []
    for vehno in tqdm(range(121, 211)):
        acc_flag = False
        acc_time = ''
        acc_speed = None
        acc_current = None
        
        data = prepare(vehno, 'test_allv2')
        battery_change = data[data['battery_pos_on'].diff() < 0]
        
        max_prob = -1
        for idx in battery_change.index:
            sample = data[max(0, idx-window): min(len(data),idx+window)].reset_index(drop = True)

            # create features
            neg_con = int(sample['battery_neg_on'].iloc[0] == 1 and sample['battery_neg_on'].iloc[-1] == 0)
            occ_con = int(sample['driver_occ_on'].iloc[0] == 1 and sample['driver_occ_on'].iloc[-1] == 0)
            spd_con = int(np.median(sample['speed'][-10:]) == 0)
            current_con = int(np.median(sample['current'][-10:]) == 0)
            low_vol_drop = int(np.min(data['low_voltage'][idx-3:idx+3].diff()[1:]) < -0.9)

            col_list = ['accelerator', 'torque', 'low_voltage', 'current', 'voltage', 'speed', 'accel', 'battery_pos_on',
                    'battery_neg_on', 
                'leaving_warn', 'seatbelt_on',
                'hand_brake_on',
                'gear_back', 'gear_parking', 'jerk', 'accelerator_diff',
        'low_voltage_diff', 'occ_diff', 'key_diff', 'gear_diff', 'brake_diff', 'run']

            des_fea = np.zeros(window*2*len(col_list))
            des_fea[:len(sample[col_list].values.reshape(-1))] = sample[col_list].values.reshape(-1)

            feature = np.append(np.array([neg_con, occ_con, spd_con, current_con, low_vol_drop]), des_fea)

            x_data = np.nan_to_num([feature], posinf = 10, neginf = -10)

            pre = model.predict(x_data)[0]
            prob = model.predict_proba(x_data)[0][1]
            
            if max_prob < prob:
                max_prob = prob

                acc_flag = True
                acc_time = data['time'][idx]
                acc_speed = data['speed'][idx]
                acc_current = data['current'][idx]
                
                acc_low_vol = data['low_voltage'][idx]
                
                if (acc_speed > 10 and acc_low_vol > 13):
                    acc_time = data['time'][idx + 1]
                    acc_speed = data['speed'][idx + 1]
                    
                
        acc_list.append([vehno, int(acc_flag), str(acc_time), max_prob, acc_speed, acc_current])
        # print([vehno, int(acc_flag), str(acc_time), max_prob])

    model1 = pd.DataFrame(acc_list, columns = ['vehno','Label','CollectTime', 'prob', 'acc_speed', 'acc_current'])

    model2 = pd.read_csv('../user_data/model_des_a_predict.csv')
    model1['prob2'] = model2['prob']
    model1['sum_prob'] = model1[['prob', 'prob2']].sum(axis = 1)
    out = model1
    out['Label'][out['sum_prob'] < 1] = 0
    out['CollectTime'][out['sum_prob'] < 1] = ''

    out = out.rename(columns = {'vehno': '车号'})
    out = out[['车号', 'Label', 'CollectTime']]
    out.to_csv('../user_data/fake_A_label.csv', index = False)

