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

def generate_train_model2():
    print('Generating train dataset for model2...')

    train_label = pd.read_csv('../raw_data/train_labels.csv', encoding = 'utf-8')
    train_label.columns = ['vehno', 'Label', 'CollectTime']
    train_label['CollectTime'][train_label['vehno'] == 77] = '2020-10-20 13:37:12' # this seems to be system error of DF

    fake_label = pd.read_csv('../user_data/fake_A_label.csv', encoding = 'utf-8')
    fake_label.columns = ['vehno', 'Label', 'CollectTime']
    com_label = pd.concat((train_label, fake_label), axis =0).reset_index(drop = True)

    window = 10

    train_x = []
    train_y = []

    for vehno in tqdm(com_label['vehno']):
        if vehno <= 120:
            data = prepare(vehno)
        else:
            data = prepare(vehno, 'test_allv2')
        
        battery_change = list(data[data['battery_pos_on'].diff() == -1].index)
        
        for idx in battery_change:
            sample = data[max(0, idx-window): min(len(data),idx+window)].reset_index(drop = True)
            
            # judege the label
            label = 0
            acc, is_acc = com_label.loc[com_label['vehno'] == vehno, 'CollectTime'].values[0], com_label.loc[com_label['vehno'] == vehno, 'Label'].values[0]
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
            
            col_list = ['accelerator', 'torque', 'low_voltage', 'current', 'voltage', 'speed', 'accel', 'battery_pos_on',
                    'battery_neg_on', 
                'leaving_warn', 'seatbelt_on',
                'hand_brake_on',
                'gear_back', 'gear_parking', 'jerk', 'accelerator_diff',
        'low_voltage_diff', 'occ_diff', 'key_diff', 'gear_diff', 'brake_diff', 'run']
            
            des_fea = np.zeros(window*2*len(col_list))
            des_fea[:len(sample[col_list].values.reshape(-1))] = sample[col_list].values.reshape(-1)
        
            feature = np.append(np.array([neg_con, occ_con, spd_con, current_con, low_vol_drop]), des_fea)
            
            train_x.append(feature)
            train_y.append(label)
            
        np.savez('../user_data/train_time_model2.npz', x = train_x, y = train_y)
            