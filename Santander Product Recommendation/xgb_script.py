'''
   author:TaoZI
   date:2016/12/22
'''
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold

pd.options.mode.chained_assignment = None

mapping_dict = {
    'sexo': {'nan': 0, 'H': 0, 'V': 1},
    'ind_actividad_cliente': {'nan': 0, '0.0': 0, '0': 0, '1.0': 1, '1': 1},
    'segmento': {'nan': 0, '01 - TOP': 0, '03 - UNIVERSITARIO': 1, '02 - PARTICULARES': 2},
    'ind_nuevo': {'nan': 0, '1.0': 0, '1': 0, '0.0': 1, '0': 1},
    'tiprel_1mes': {'nan': 0, 'P': 0, 'R': 0, 'N': 0, 'I': 1, 'A': 2},
    'indext': {'nan': 0, 'S': 0, 'N': 1}
}

target_raw_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                   'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                   'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                   'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                   'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

target_cols = target_raw_cols[2:]

con_cols = ['ncodpers', 'fecha_dato', 'age', 'antiguedad', 'renta']
cat_cols = mapping_dict.keys()
user_cols = con_cols + cat_cols + target_raw_cols
NUM_CLASS = 22


def getAge(str_age):
    age = str_age.strip()
    if age == 'NA' or age == 'nan':
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


def getCustSeniority(str_seniority):
    cust_seniority = str_seniority.strip()
    if cust_seniority == 'NA' or cust_seniority == 'nan':
        seniority = 4
    elif float(cust_seniority) < 50:
        seniority = 0
    elif float(cust_seniority) < 75:
        seniority = 1
    elif float(cust_seniority) < 100:
        seniority = 2
    elif float(cust_seniority) < 125:
        seniority = 3
    elif float(cust_seniority) < 150:
        seniority = 4
    elif float(cust_seniority) < 175:
        seniority = 5
    elif float(cust_seniority) < 200:
        seniority = 6
    elif float(cust_seniority) < 225:
        seniority = 7
    else:
        seniority = 8
    return seniority


def getRent(str_rent):
    rent = str_rent.strip()
    if rent == 'NA' or rent == 'nan':
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


def add_com_features(lag_feats):
    lag_feats['prod_sum'] = lag_feats.apply(lambda x: np.sum(x[-120:]), axis=1)

    for i, pre in enumerate(['1_', '2_', '3_', '4_', '5_']):
        pre_cols = [pre + col for col in target_raw_cols]
        lag_feats['sum_24_' + str(i + 1)] = lag_feats.loc[:, pre_cols].sum(axis=1)
    sum_24_list = ['sum_24_' + str(i + 1) for i in range(5)]
    lag_feats['sum_24_max'] = lag_feats[sum_24_list].max(axis=1)
    lag_feats['sum_24_min'] = lag_feats[sum_24_list].min(axis=1)
    lag_feats['sum_24_mean'] = lag_feats[sum_24_list].mean(axis=1)

    for i, col in enumerate(target_raw_cols):
        index_list = [pre + col for pre in ['1_', '2_', '3_', '4_', '5_']]
        lag_feats['prod_sum_' + str(i)] = lag_feats.loc[:, index_list].sum(axis=1)

    pro_sum_list = ['prod_sum_' + str(i) for i in range(24)]
    for gp_col in ['renta', 'sexo']:
        group_feats = lag_feats[pro_sum_list].groupby(lag_feats[gp_col]).agg(lambda x: round(x.sum() / x.count(), 2))
        group_feats.columns = [gp_col + str(i) for i in range(24)]
        lag_feats = pd.merge(lag_feats, group_feats, left_on=gp_col, right_index=True, how='left')

    com_col = [[0, 2], [7, 8, 9], [9, 10, 11], [19, 20, 21]]
    for x in range(4):
        import_col = [target_cols[i] for i in com_col[x]]
        for i in range(1, 6):
            pre_import_col = [str(i) + '_' + col for col in import_col]
            lag_feats[str(i) + '_' + str(x + 1) + '_s_sum_import'] = lag_feats[pre_import_col].sum(axis=1)
    return lag_feats


def process_train_data(in_file_name, date_list):
    this_month = in_file_name[in_file_name['fecha_dato'].isin([date_list[0]])]
    for col in cat_cols:
        this_month[col] = this_month[col].apply(lambda x: mapping_dict[col][str(x)])
    for col in target_raw_cols:
        this_month[col].fillna(0, inplace=True)
    this_month['age'] = this_month['age'].apply(lambda x: getAge(x))
    this_month['antiguedad'] = this_month['antiguedad'].apply(lambda x: getCustSeniority(x))
    this_month['renta'] = this_month['renta'].apply(lambda x: getRent(str(x)))

    hist_data = in_file_name.loc[:, ['ncodpers', 'fecha_dato'] + target_raw_cols]
    del in_file_name
    pre_month = hist_data[hist_data['fecha_dato'].isin([date_list[1]])]
    pre_month_ncodpers = pre_month[['ncodpers']]
    pre_month_target = pre_month[target_raw_cols]
    pre_month_target = pre_month_target.add_prefix('1_')
    pre_month = pd.concat([pre_month_ncodpers, pre_month_target], axis=1)
    this_month = pd.merge(this_month, pre_month, on=['ncodpers'], how='left')
    this_month.fillna(0, inplace=True)
    for col in target_cols:
        this_month[col] = np.where(this_month[col] - this_month['1_' + col] > 0,
                                   (this_month[col] - this_month['1_' + col]), 0)

    this_month_target = this_month[target_cols]
    this_month = this_month.drop(target_raw_cols, axis=1)

    x_vars_list = []
    y_vars_list = []

    for i in range(2, len(date_list)):
        tmp = hist_data[hist_data['fecha_dato'].isin([date_list[i]])].loc[:, ['ncodpers'] + target_raw_cols]
        tmp = tmp.add_prefix(str(i) + "_")
        tmp.rename(columns={str(i) + '_ncodpers': 'ncodpers'}, inplace=True)
        this_month = pd.merge(this_month, tmp, on=['ncodpers'], how='left')
    this_month.fillna(0, inplace=True)
    del hist_data

    this_month = add_com_features(this_month)
    this_month.fillna(0, inplace=True)

    this_month = pd.concat([this_month, this_month_target], axis=1)
    for idx, row in this_month.iterrows():
        for i in range(0, 22):
            if row[(-22 + i)] > 0:
                x_vars_list.append(row[:-22])
                y_vars_list.append(i)

    return np.array(x_vars_list), np.array(y_vars_list)


def process_test_data(test_file, hist_file, date_list):
    for col in cat_cols:
        test_file[col] = test_file[col].apply(lambda x: mapping_dict[col][str(x)])
    test_file['age'] = test_file['age'].apply(lambda x: getAge(x))
    test_file['antiguedad'] = test_file['antiguedad'].apply(lambda x: getCustSeniority(x))
    test_file['renta'] = test_file['renta'].apply(lambda x: getRent(x))

    for i in range(0, len(date_list)):
        tmp = hist_file[hist_file['fecha_dato'].isin([date_list[i]])].loc[:, ['ncodpers'] + target_raw_cols]
        tmp = tmp.add_prefix(str(i + 1) + "_")
        tmp.rename(columns={str(i + 1) + '_ncodpers': 'ncodpers'}, inplace=True)
        test_file = pd.merge(test_file, tmp, on=['ncodpers'], how='left')
    test_file.fillna(0, inplace=True)

    del hist_file

    test_file = add_com_features(test_file)
    test_file.fillna(0, inplace=True)
    return test_file.values


def runXGB_CV(train_X, train_y, test_X, index, seed_val):
    train_index, test_index = index
    X_train = train_X[train_index]
    y_train = train_y[train_index]

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(test_X)

    param = {
        'objective': 'multi:softprob',
        'eval_metric': "mlogloss",
        'num_class': NUM_CLASS,
        'silent': 1,
        'min_child_weight': 2,
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'seed': seed_val
    }
    num_rounds = 100
    model = xgb.train(param, xgtrain, num_rounds)
    pred = model.predict(xgtest)
    return pred


def runXGB(train_X, train_y, test_X, seed_val=123):
    param = {
        'objective': 'multi:softprob',
        'eval_metric': "mlogloss",
        'num_class': NUM_CLASS,
        'silent': 1,
        'min_child_weight': 2,
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'seed': seed_val
    }
    num_rounds = 100
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X)

    model = xgb.train(param, xgtrain, num_rounds)
    preds = model.predict(xgtest)
    return preds


if __name__ == "__main__":

    cv_sel = 1
    start_time = datetime.datetime.now()
    data_path = '../input/'

    print "feature extract..."
    train_file = pd.read_csv(data_path + 'train_ver3.csv',
                             dtype={'age': 'str', 'antiguedad': 'str', 'renta': 'str'},
                             usecols=user_cols)
    print datetime.datetime.now() - start_time

    train_X, train_y = process_train_data(train_file, ['2015-06-28', '2015-05-28', '2015-04-28',
                                                       '2015-03-28', '2015-02-28', '2015-01-28'])
    train_X = train_X[:, 2:]
    print datetime.datetime.now() - start_time

    data_date = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28']
    train_file = train_file[train_file['fecha_dato'].isin(data_date)].loc[:,
                 ['ncodpers', 'fecha_dato'] + target_raw_cols]

    test_file = pd.read_csv(data_path + 'test_ver3.csv',
                            dtype={'age': 'str', 'antiguedad': 'str', 'renta': 'str'},
                            usecols=con_cols + cat_cols)

    test_X = process_test_data(test_file, train_file, data_date)
    print datetime.datetime.now() - start_time

    del train_file, test_file
    test_X = test_X[:, 2:]
    feats = feats[2:]
    print train_X.shape, train_y.shape, test_X.shape
    print datetime.datetime.now() - start_time

    seed_val = 123
    if cv_sel == 1:
        print "running model with cv..."
        nfolds = 5
        kf = KFold(train_X.shape[0], n_folds=nfolds, shuffle=True, random_state=seed_val)
        preds = [0] * NUM_CLASS
        for i, index in enumerate(kf):
            preds += runXGB_CV(train_X, train_y, test_X, index, seed_val)
            print 'fold %d' % (i + 1)
        preds = preds / nfolds

    else:
        print "running model with feature..."
        preds = runXGB(train_X, train_y, test_X, seed_val)

    del train_X, test_X, train_y

    print "Getting the top products.."
    target_cols = np.array(target_cols)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :7]
    test_id = np.array(pd.read_csv(data_path + 'test_ver2.csv', usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv('../submit/sub_xgb.csv', index=False)
