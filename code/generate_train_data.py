#!/usr/bin/env python
# coding: utf-8
import pdb

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def prepare(vehno, data_name = 'train'):
    data = pd.read_csv(f'../raw_data/{data_name}/{vehno}.csv', encoding = 'utf-8')
    
    data.columns = ['veh_id', 'time', 'accelerator', 'battery_pos', 'battery_neg', 'brake', 
                   'leaving', 'driver_occ', 'seatbelt', 'torque', 'hand_brake', 'key_status', 
                   'low_voltage', 'gear', 'current', 'voltage', 'mile', 'speed', 'steering']

    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values(by = 'time').reset_index(drop = True)
    time_diff = data['time'].diff().apply(lambda x: pd.Timedelta(x).seconds).values
    speed_diff = data['speed'].diff().values/3.6
    data['accel'] = speed_diff/time_diff

    data['battery_pos'][:2] = ['断开', '连接']
    data['battery_neg'][:3] =  ['断开', '粘连', '连接']
    data['brake'][:2] = ['未踩', '踩下']
    data['leaving'][:2]= ['No Warning', 'Warning(下车时请拔掉钥匙)']
    data['driver_occ'][:3] = ['传感器故障', '有人', '空置']
    data['seatbelt'][:2]= ['已系', '未系']
    data['hand_brake'][:2]= ['手刹拉起', '手刹放下']
    data['key_status'][:3]= ['ACC', 'OFF', 'ON']
    data['gear'][:4]=['前进', '后退', '空档', '驻车']

    data = pd.get_dummies(data, dtype = int)
    data = data[5:].reset_index(drop = True)

    data.columns = ['veh_id', 'time', 'accelerator', 'torque', 'low_voltage', 'current',
           'voltage', 'mile', 'speed', 'steering', 'accel', 'battery_pos_off',
           'battery_pos_on', 'battery_neg_off', 'battery_neg_mid', 'battery_neg_on',
           'brake_on', 'brake_off', 'leaving_no_warn',
           'leaving_warn', 'driver_occ_sensor', 'driver_occ_on',
           'driver_occ_off', 'seatbelt_on', 'seatbelt_off', 'hand_brake_on',
           'hand_brake_off', 'key_ACC', 'key_off', 'key_on',
           'gear_forward', 'gear_back', 'gear_null', 'gear_parking']

    data['run'] = data['battery_pos_on'] + data['battery_neg_on'] + data['leaving_no_warn'] + data['driver_occ_on'] + data['seatbelt_on'] + data['hand_brake_off'] + data['key_on'] + data['gear_forward']

    data = data.drop(['battery_pos_off', 'battery_neg_off', 'leaving_no_warn', 'driver_occ_sensor',
                      'seatbelt_off', 'key_ACC', 'gear_null', 'brake_off'], axis = 1)

    data['jerk'] = data['accel'].diff()
    data['accelerator_diff'] = data['accelerator'].diff()
    data['low_voltage_diff'] = data['low_voltage'].diff()
    data['occ_diff'] = data['driver_occ_on'].diff()
    data['key_diff'] = data['key_on'].diff()
    data['gear_diff'] = data['gear_forward'].diff()
    data['brake_diff'] = data['brake_on'].diff()
    return data

def generate_train():
    print('Generating train dataset...')

    train_label = pd.read_csv('../raw_data/train_labels.csv', encoding = 'utf-8')
    train_label.columns = ['vehno', 'Label', 'CollectTime']
    train_label['CollectTime'][train_label['vehno'] == 77] = '2020-10-20 13:37:12'

    window = 10

    train_x = []
    train_y = []

    for vehno in tqdm(range(1, 121)):
        data = prepare(vehno)
        
        battery_change = list(data[data['battery_pos_on'].diff() == -1].index)
        
        for idx in battery_change:
            sample = data[max(0, idx-window): min(len(data), idx + window)].reset_index(drop = True)
            
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
            
            col_list = ['accelerator', 'torque', 'low_voltage', 'current', 'voltage', 'speed', 'accel', 'battery_pos_on',
                    'battery_neg_on', 
                'leaving_warn', 'seatbelt_on',
                'hand_brake_on',
                'gear_back', 'gear_parking', 'jerk', 'accelerator_diff',
        'low_voltage_diff', 'occ_diff', 'key_diff', 'gear_diff', 'brake_diff', 'run']
            
            des_fea = sample[col_list].values.reshape(-1)
        
            feature = np.append(np.array([neg_con, occ_con, spd_con, current_con, low_vol_drop]), des_fea)
            
            train_x.append(feature)
            train_y.append(label)
            
        np.savez('../user_data/train_time.npz', x = train_x, y = train_y)
