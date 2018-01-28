import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import itertools
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

pd.options.mode.chained_assignment = None

multi_corr = [79, 80, 81, 87, 89, 90, 101, 103, 111]
two_corr = [2, 3, 9, 10, 11, 12, 13, 23, 36, 57, 72]
multi_cat_diff = [90, 92, 96, 99, 101, 102, 103, 106, 109, 110, 113, 114, 116]
skewed_num = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
cat2corr = [(29, 30), (40, 41), (43, 45), (55, 56), (8, 65), (8, 66), (104, 106)]
two_avg1 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 23, 24, 25, 26, 27, 28, 36, 38, 40, 44, 50, 53, 57, 72, 73,
            76, 79, 80, 81, 82, 87, 89, 90, 103, 111]


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con ** 2 / (np.abs(x) + con) ** 2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A'))
    return r + 1


def prepro(train, test, cont_feature):
    joined = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = ['cont' + str(i) for i in skewed_num]
    for feats in skewed_feats:
        joined[feats] = joined[feats] + 1
        joined[feats], lam = boxcox(joined[feats])

    multi_diff_feats = ['cat' + str(i) for i in multi_cat_diff]
    for column in multi_diff_feats:
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train
        remove = remove_train.union(remove_test)

        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)

    ss = StandardScaler()
    joined[cont_feature] = ss.fit_transform(joined[cont_feature].values)
    del train, test
    return joined


def feature_extract(joined, cont_feature):
    features = pd.DataFrame()
    features['id'] = joined['id']
    features['loss'] = np.log(joined['loss'] + 200)

    cat_sel = [n for n in joined.columns if n.startswith('cat')]
    for column in cat_sel:
        features[column] = pd.factorize(joined[column].values, sort=True)[0] + 1

    for column in cont_feature:
        features[column] = joined[column]

    features['cont_avg'] = joined[cont_feature].mean(axis=1)
    features['cont_min'] = joined[cont_feature].min(axis=1)
    features['cont_max'] = joined[cont_feature].max(axis=1)

    for i in [20, 40, 73]:
        cat_feats = ['cat' + str(i) for i in range(1, i)]
        idx = 'cat_' + 'sum_' + str(i)
        features[idx + '_A'] = joined[cat_feats].apply(lambda x: sum(x == 'A'), axis=1)
        features[idx + '_B'] = joined[cat_feats].apply(lambda x: sum(x == 'B'), axis=1)

    cat2_feats = [('cat' + str(i), 'cat' + str(j)) for (i, j) in cat2corr]
    for feat1, feat2 in cat2_feats:
        feat_comb = feat1 + '_' + feat2
        features[feat_comb] = joined[feat1] + joined[feat2]
        features[feat_comb] = features[feat_comb].apply(encode)

    cat2avg_feats = ['cat' + str(i) for i in two_avg1]
    for feat1, feat2 in itertools.combinations(cat2avg_feats, 2):
        feat_comb = feat1 + '_' + feat2
        features[feat_comb] = joined[feat1] + joined[feat2]
        features[feat_comb] = features[feat_comb].apply(encode)

    train = features[features['loss'].notnull()]
    test = features[features['loss'].isnull()]
    del features, joined
    return train, test


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def feature_select(train, test):
    import operator
    params = {
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'objective': 'reg:linear',
        'verbose_eval': True,
        'seed': 12
    }
    rounds = 300
    y = train['loss']
    X = train.drop(['loss', 'id'], 1)

    xgtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)

    feats = [x for x in train.columns if x not in ['id', 'loss']]
    print len(feats)
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in feats:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    feats = [a for (a, b) in importance]
    feats = feats[:450]
    print len(feats)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("../input/feat_sel/feat_importance.csv", index=False)

    train1 = train[['id', 'loss'] + feats]
    test1 = test[['id'] + feats]
    return train1, test1


def runXGB(train, test, index, RANDOM_STATE):
    train_index, test_index = index
    y = train['loss']
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['id'], 1)
    del train, test
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_val, label=y_val)
    xgtest = xgb.DMatrix(X_test)
    X_val = xgb.DMatrix(X_val)

    params = {
        'min_child_weight': 10,
        'eta': 0.01,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }
    rounds = 3000

    watchlist = [(xgtrain, 'train'), (xgval, 'eval')]
    model = xgb.train(params, xgtrain, rounds, watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=100)

    cv_score = mean_absolute_error(np.exp(model.predict(X_val)) - 200, np.exp(y_val) - 200)
    predict = np.exp(model.predict(xgtest)) - 200
    print "iteration = %d" % (model.best_iteration)
    return predict, cv_score


if __name__ == '__main__':

    Generate_or_read = 0  # 0 generate
    feat_sel = 1  # 1 select
    start_time = datetime.datetime.now()
    if Generate_or_read == 0:
        print "generate features..."
        train = pd.read_csv('../input/train.csv')
        test = pd.read_csv('../input/test.csv')
        test['loss'] = np.nan
        cont_feature = [n for n in train.columns if n.startswith('cont')]
        joined = prepro(train, test, cont_feature)
        train, test = feature_extract(joined, cont_feature)
        print train.shape, test.shape
        print datetime.datetime.now() - start_time
        if feat_sel == 1:
            print "feature select..."
            train, test = feature_select(train, test)
        train.to_csv("../input/feature/train.csv", index=False)
        test.to_csv("../input/feature/test.csv", index=False)
        print train.shape, test.shape
        print datetime.datetime.now() - start_time

    else:
        print "read features..."
        train = pd.read_csv("../input/feature/train.csv")
        test = pd.read_csv("../input/feature/test.csv")
        print train.shape, test.shape

    print "run model..."
    nfolds = 10
    RANDOM_STATE = 113
    ids = test['id']
    predicts = np.zeros(ids.shape)
    kf = KFold(train.shape[0], n_folds=nfolds, shuffle=True, random_state=RANDOM_STATE)
    for i, index in enumerate(kf):
        print('Fold %d' % (i + 1))
        predict, cv_score = runXGB(train, test, index, RANDOM_STATE)
        print cv_score
        predicts += predict

    print datetime.datetime.now() - start_time
    predicts = predicts / nfolds
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['loss'] = predicts
    submission.to_csv('../submit/submit_xgb.csv', index=False)
