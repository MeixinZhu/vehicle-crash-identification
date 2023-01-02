import pandas as pd
from sklearn.metrics import f1_score
import pdb

def combine():
    print('Combine the results of model 1 and 2, generate final results...')
    model1 = pd.read_csv('../user_data/model1.csv')
    model2 = pd.read_csv('../user_data/model2.csv')


    model1['prob2'] = model2['prob']
    model1['max_prob'] = model1[['prob', 'prob2']].max(axis = 1)
    out = model1
    out['Label'][out['max_prob'] < 0.5] = 0
    out['CollectTime'][out['max_prob'] < 0.5] = ''

    out = out.rename(columns = {'vehno': '车号'})
    out = out[['车号', 'Label', 'CollectTime']]

    # test_predict = pd.read_csv('../user_data/out_B_final_v2.csv')
    # print(f1_score(test_predict['Label'], out['Label']))

    out.to_csv('../prediction_result/result.csv', index = False)
    print(out)
    print('Total predicted events:', out['Label'].sum())
    # print(out[out['Label']!= test_predict['Label']])
    # pdb.set_trace()
