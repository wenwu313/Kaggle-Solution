import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold

target_raw_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                   'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                   'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                   'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                   'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

target_cols = target_raw_cols[2:]
NUM_CLASS = 22


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
        'eta': 0.06,
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
        'eta': 0.06,
        'max_depth': 8,
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
    cv_sel = 0
    print 'read files...'
    data_path = '../input/feats/'

    columns = ['age', 'antiguedad', 'renta',
               'sexo', 'ind_actividad_cliente', 'segmento', 'ind_nuevo', 'tiprel_1mes', 'indext']
    train_X = pd.read_csv(data_path + 'train_feats_users.csv', usecols=columns + ['label'])
    for col in ['sum', 'renta', 'canel', 'lag5', 'com20', 'sum8']:
        train_temp = pd.read_csv(data_path + 'train_feats_' + col + '.csv')
        train_X = pd.concat([train_X, train_temp], axis=1)
    del train_temp

    test_X = pd.read_csv(data_path + 'test_feats_users.csv', usecols=columns)
    for col in ['sum', 'renta', 'canel', 'lag5', 'com20', 'sum8']:
        test_temp = pd.read_csv(data_path + 'test_feats_' + col + '.csv')
        test_X = pd.concat([test_X, test_temp], axis=1)
    del test_temp

    # train_X  = pd.read_csv(data_path + 'train_feats_v1.csv')
    # test_X  = pd.read_csv(data_path + 'test_feats_v1.csv')

    train_y = train_X['label'].values
    train_X = train_X.drop('label', axis=1).values
    test_X = test_X.values
    print train_X.shape, train_y.shape, test_X.shape

    seed_val = 1234
    if cv_sel == 1:
        print "running model with cv..."
        nfolds = 10
        kf = KFold(train_X.shape[0], n_folds=nfolds, shuffle=True, random_state=seed_val)
        preds = [0] * NUM_CLASS
        for i, index in enumerate(kf):
            preds += runXGB_CV(train_X, train_y, test_X, index, seed_val)
            print 'fold %d' % (i + 1)
        preds = preds / nfolds

    else:
        print "running model..."
        preds = runXGB(train_X, train_y, test_X, seed_val=seed_val)

    print "Getting the top products..."
    target_cols = np.array(target_cols)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :7]
    test_id = np.array(pd.read_csv('../input/test_ver2.csv', usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv('../submit/sub_xgb.csv', index=False)
