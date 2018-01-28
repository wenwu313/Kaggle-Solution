import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

SHIFT = 200


def df_cleaner(df_train, df_test):
    cont_list = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10',
                 'cont11', 'cont12', 'cont13', 'cont14']

    ntrain = df_train.shape[0]
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    df_cat = pd.get_dummies(df.filter(regex="^cat"))
    scale = StandardScaler()
    df[cont_list] = scale.fit_transform(df[cont_list].values)

    df = pd.concat([df[['id', 'loss'] + cont_list], df_cat], axis=1)
    df_out_train = df.iloc[:ntrain, :]
    df_out_test = df.iloc[ntrain:, :]

    df_out_columns = df_out_train.loc[:, (df_out_train != 0).any(axis=0)].columns
    data_columns = list(df_out_columns)
    data_columns.remove('id')
    data_columns.remove('loss')
    return df_out_train, df_out_test, data_columns


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds) - SHIFT, np.exp(labels) - SHIFT)


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    df_cleaner(df_train, df_test)
    train, test, features = df_cleaner(df_train, df_test)
    del df_train
    del df_test

    x_test = test[:][features]
    train['loss_logshift'] = np.log(train['loss'] + SHIFT)

    number_of_bagging_iterations = 10
    max_number_of_rounds = 1500
    early_stopping_rounds = 20

    work_dataframe = test[['id']]

    for i in xrange(number_of_bagging_iterations):
        train_slice = train[train.id % number_of_bagging_iterations != i]
        val_slice = train[train.id % number_of_bagging_iterations == i]

        x_train = train_slice[features]
        y_train = train_slice['loss_logshift']

        x_val = val_slice[features]
        y_val = val_slice['loss_logshift']

        model = xgb.XGBRegressor(max_depth=12, colsample_bytree=0.5, min_child_weight=1, subsample=0.8, gamma=1,
                                 n_estimators=max_number_of_rounds, learning_rate=0.1)

        model.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds,
                  eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric=evalerror)

        this_iteration_predictions = model.predict(x_test).astype(float)

        temp_series = pd.Series(np.exp(this_iteration_predictions) - SHIFT)
        work_dataframe['round' + str(i)] = temp_series.values

    work_dataframe['mean_values'] = work_dataframe.filter(regex="^round").mean(axis=1)
    work_dataframe[['id', 'mean_values']].to_csv('../input/submit_claim.csv', index=False,
                                                 float_format='%.2f', header=['id', 'loss'])
