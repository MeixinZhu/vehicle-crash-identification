#!/usr/bin/env python
# coding: utf-8
import pdb

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
from generate_train_data import prepare
warnings.filterwarnings('ignore')

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
fc_parameters = {
"c3": [{"lag": lag} for lag in [2]],
    
"augmented_dickey_fuller": [{"attr": "teststat"}],

"energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 4}],

"fourier_entropy":  [{"bins": x} for x in [ 20]],
"permutation_entropy":  [{"tau": 1, "dimension": x} for x in [3, 5, 7]]}


def generate_train_des():
    print('Generating train dataset...')

    train_label = pd.read_csv('../raw_data/train_labels.csv', encoding = 'utf-8')
    train_label.columns = ['vehno', 'Label', 'CollectTime']
    train_label['CollectTime'][train_label['vehno'] == 77] = '2020-10-20 13:37:12'

    window = 40
    train_x = []
    train_y = []

    for vehno in tqdm(range(1, 121)):
        data = prepare(vehno)
        
        battery_change = list(data[data['battery_pos_on'].diff() == -1].index)
        
        for idx in battery_change:
            sample = data[idx-window: idx+window].reset_index(drop = True)

            # judege the label
            label = 0
            acc, is_acc = train_label['CollectTime'][vehno - 1] , train_label['Label'][vehno - 1]
            if is_acc == 1:
                acc_time = datetime.strptime(acc, "%Y-%m-%d %H:%M:%S")
                if sample['time'].iloc[0] < acc_time and acc_time < sample['time'].iloc[-1]:
                    label = 1
                    
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
            
            train_x.append(feature)
            train_y.append(label)

    np.savez('../user_data/train_des.npz', x = train_x, y = train_y)
