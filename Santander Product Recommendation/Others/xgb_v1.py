
import csv
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


mapping_dict = {
'sexo'          : {-99:0, 'H':0, 'V':1},
'ind_actividad_cliente' : {-99:0, '0.0':0, '0':0,'1.0':1, '1':1},
'segmento'      : {-99:0, '01 - TOP':0, '03 - UNIVERSITARIO':1, '02 - PARTICULARES':2},
'ind_nuevo'     : {-99:0,  '1.0':0, '1':0,  '0.0':1, '0':1 },
'tiprel_1mes'   : {-99:0,  'P':0, 'R':0, 'N':0, 'I':1, 'A':2},
'indext'        : {-99:0,  'S':0, 'N':1},
# 'canal_entrada' : {'KHE':6, 'KAT':5 ,'KFC':4, 'KFA':3, 'KHK':2, 'KHQ':1, -99: 0}
}
target_cols1 = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',  'ind_cder_fin_ult1',
                'ind_cno_fin_ult1',  'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',  'ind_plan_fin_ult1',
                'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                'ind_viv_fin_ult1',  'ind_nomina_ult1',   'ind_nom_pens_ult1', 'ind_recibo_ult1']

cat_cols = list(mapping_dict.keys())
target_cols = target_cols1[2:]
target_cols.remove('ind_cder_fin_ult1')
NUM_CLASS = 22

def getIndex(row, col):
    val = row[col].strip()
    if val not in ['', 'NA']:
        ind = mapping_dict[col][val]
    else:
        ind = mapping_dict[col][-99]
    return ind


def getAge(row):
    age = row['age'].strip()
    if age == 'NA' or age == '':
        age1 = 2
    elif float(age) < 20:
        age1 = 0
    elif float(age) < 30:
        age1 = 1
    elif float(age) < 40:
        age1 = 2
    elif float(age) < 50:
        age1 = 3
    elif float(age) < 60:
        age1 = 4
    else:
        age1 = 5
    return age1

def getCustSeniority(row):
    cust_seniority = row['antiguedad'].strip()
    if cust_seniority == 'NA' or cust_seniority == '':
        seniority = 2
    elif float(cust_seniority) < 50:
        seniority = 0
    elif float(cust_seniority) < 100:
        seniority = 1
    elif float(cust_seniority) < 150:
        seniority = 2
    elif float(cust_seniority) < 200:
        seniority = 3
    else:
        seniority = 4
    return seniority

def getRent(row):
    rent = row['renta'].strip()
    if rent == 'NA' or rent == '':
        rent1 = 4
    elif float(rent) < 45542.97:
        rent1 = 1
    elif float(rent) < 57629.67:
        rent1 = 2
    elif float(rent) < 68211.78:
        rent1 = 3
    elif float(rent) < 78852.39:
        rent1 = 4
    elif float(rent) < 90461.97:
        rent1 = 5
    elif float(rent) < 103855.23:
        rent1 = 6
    elif float(rent) < 120063.00:
        rent1 = 7
    elif float(rent) < 141347.49:
        rent1 = 8
    elif float(rent) < 173418.12:
        rent1 = 9
    elif float(rent) < 234687.12:
        rent1 = 10
    else:
        rent1 = 11
    return rent1

def getTarget(row):
    tlist = []
    for col in target_cols:
        if row[col].strip() in ['', 'NA']:
            target = 0
        else:
            target = int(float(row[col]))
        tlist.append(target)
    print len(tlist)
    return tlist

def feature_extract(row, prev_target_list):
    analy_index = [0,1,8,9,10,13,14,15,16,18,19,20]
    pro_feats = [prev_target_list[i] for i in analy_index]
    x_vars = []
    for col in cat_cols:
        x_vars.append(getIndex(row, col))
    x_vars.append(getAge(row))
    x_vars.append(getCustSeniority(row))
    x_vars.append(getRent(row))
    x_vars.append(prev_target_list.count(1))
    return x_vars + pro_feats

def getLagFeature():
    data_path = '../input/divide/train'
    use_cols = ['ncodpers'] + target_cols1
    train_05 = pd.read_csv(data_path + '2015-05-28.csv',usecols = use_cols)
    train_04 = pd.read_csv(data_path + '2015-04-28.csv', usecols=use_cols)
    train_03 = pd.read_csv(data_path + '2015-03-28.csv', usecols=use_cols)
    train_02 = pd.read_csv(data_path + '2015-02-28.csv', usecols=use_cols)
    train_01 = pd.read_csv(data_path + '2015-01-28.csv', usecols=use_cols)
    train_lag = pd.merge(train_05,train_04, on = 'ncodpers',how = 'left')
    train_lag = pd.merge(train_lag, train_03, on = 'ncodpers', how = 'left')
    train_lag = pd.merge(train_lag, train_02, on = 'ncodpers', how = 'left')
    train_lag = pd.merge(train_lag, train_01, on = 'ncodpers', how = 'left')
    train_lag.fillna(0 ,inplace = True)
    train_lag_dict = {}
    for ind, row in train_lag.iterrows():
        id = int(row['ncodpers'])
        train_lag_dict[id] = list(row.values[1:])

    train_05 = pd.read_csv(data_path + '2016-05-28.csv', usecols=use_cols)
    train_04 = pd.read_csv(data_path + '2016-04-28.csv', usecols=use_cols)
    train_03 = pd.read_csv(data_path + '2016-03-28.csv', usecols=use_cols)
    train_02 = pd.read_csv(data_path + '2016-02-28.csv', usecols=use_cols)
    train_01 = pd.read_csv(data_path + '2016-01-28.csv', usecols=use_cols)
    train_lag = pd.merge(train_05, train_04, on='ncodpers', how='left')
    train_lag = pd.merge(train_lag, train_03, on='ncodpers', how='left')
    train_lag = pd.merge(train_lag, train_02, on='ncodpers', how='left')
    train_lag = pd.merge(train_lag, train_01, on='ncodpers', how='left')
    train_lag.fillna(0, inplace=True)
    test_lag_dict = {}
    for ind, row in train_lag.iterrows():
        id = int(row['ncodpers'])
        test_lag_dict[id] = list(row.values[1:])
    return train_lag_dict,test_lag_dict

def getTrainTestSet():
    train_lag_dict, test_lag_dict = getLagFeature()
    x_vars_list = []
    y_vars_list = []
    data_path = '../input/divide/train'

    train_file = open(data_path  + '2015-05-28.csv')
    cust_dict = {}
    for row in csv.DictReader(train_file):
        cust_id = int(row['ncodpers'])
        cust_dict[cust_id] = getTarget(row)
    train_file.close()

    train_file = open(data_path + '2015-06-28.csv')
    for row in csv.DictReader(train_file):
        cust_id = int(row['ncodpers'])
        prev_target_list = cust_dict.get(cust_id, [0] * NUM_CLASS)
        target_list = getTarget(row)
        new_products = [max(x1 - x2, 0) for (x1, x2) in zip(target_list, prev_target_list)]
        if sum(new_products) > 0:
            for ind, prod in enumerate(new_products):
                if prod > 0:
                    x_vars = feature_extract(row, prev_target_list)
                    x_vars.extend(train_lag_dict.get(cust_id, [0] * 120))
                    x_vars_list.append(x_vars)
                    y_vars_list.append(ind)
    train_file.close()

    test_file = open(data_path + '2016-05-28.csv')
    cust_dict = {}
    for row in csv.DictReader(test_file):
        cust_id = int(row['ncodpers'])
        cust_dict[cust_id] = getTarget(row)
    test_file.close()

    x_test_list = []
    test_file = open('../input/test_ver2.csv')
    for row in csv.DictReader(test_file):
        cust_id = int(row['ncodpers'])
        prev_target_list = cust_dict.get(cust_id, [0] * NUM_CLASS)
        x_vars = feature_extract(row, prev_target_list)
        x_vars.extend(test_lag_dict.get(cust_id, [0] * 120))
        x_test_list.append(x_vars)
    test_file.close()

    train_X = np.array(x_vars_list)
    train_y = np.array(y_vars_list)
    test_X = np.array(x_test_list)

    print train_X.shape, train_y.shape, test_X.shape
    return train_X, train_y, test_X

def runXGB(xgtrain, seed_val=123):
    param = {
        'objective' : 'multi:softprob',
        'eval_metric' : "mlogloss",
        'num_class' : NUM_CLASS,
        'silent' : 1,
        'min_child_weight' : 2,
        'eta': 0.05,
        'max_depth': 6,
        'subsample' : 0.9,
        'colsample_bytree' : 0.8,
        'seed' : seed_val
    }
    num_rounds = 100
    model = xgb.train(param, xgtrain, num_rounds)
    return model

if __name__ == "__main__":

    print "feature extract..."
    start_time = datetime.datetime.now()
    train_X, train_y, test_X = getTrainTestSet()

    xgtrain = xgb.DMatrix(train_X, label = train_y)
    xgtest  = xgb.DMatrix(test_X)
    xgval = xgb.DMatrix(train_X)
    y_true = train_y
    del train_X, train_y, test_X
    print(datetime.datetime.now() - start_time)

    print "running model..."
    model = runXGB(xgtrain, seed_val=123)
    y_pred = model.predict(xgval)
    print log_loss(y_true, y_pred)

    preds = model.predict(xgtest)
    del xgtrain, xgtest
    print(datetime.datetime.now() - start_time)

    print "Getting the top products.."
    target_cols = np.array(target_cols)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :8]
    test_id = np.array(pd.read_csv("../input/test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv('../submit/sub_xgb.csv', index=False)
    print(datetime.datetime.now() - start_time)