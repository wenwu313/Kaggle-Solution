sub_rf = pd.read_csv('../input/sub_rf.csv', nrows=929615)
sub_reg = pd.read_csv('../input/sub_reg.csv', nrows=929615)
sub_union = pd.DataFrame(np.zeros((sub_rf.shape[0], 2)), columns=['ncodpers', 'added_products'])
sub_union['ncodpers'] = sub_rf['ncodpers']

added = []
for x in range(sub_rf.shape[0]):
    rf_str = sub_rf.loc[x]['added_products'].split(' ')
    reg_str = sub_reg.loc[x]['added_products'].split(' ')[:7]

    str = []
    for str1 in reg_str:
        if str1 in rf_str:
            str.append(str1)
    for str1 in rf_str:
        if str1 not in str:
            str.append(str1)
    added.append(" ".join(str))

sub_union.loc[:, 'added_products'] = added
sub_union.to_csv('submit.csv', index=False)


# ======================================================================================================================
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


# df_user['renta'] = df_user['renta'].fillna(df_user.loc[df_user['renta'].notnull(),'renta'].median())
# ======================================================================================================================
for con_attr in ['age', 'antiguedad', 'renta']:
    group_feats_1 = lag_feats[pro_sum_list].groupby(lag_feats[con_attr]).agg(lambda x: x.sum())
    group_feats_0 = lag_feats[pro_sum_list].groupby(lag_feats[con_attr]).agg(lambda x: x.count() - x.sum())
    group_feats_r = lag_feats[pro_sum_list].groupby(lag_feats[con_attr]).agg(lambda x: round(x.sum() / x.count(), 2))
    group_feats_1.columns = [con_attr + '_1_' + str(i) for i in range(24)]
    group_feats_0.columns = [con_attr + '_0_' + str(i) for i in range(24)]
    group_feats_r.columns = [con_attr + '_r_' + str(i) for i in range(24)]
    lag_feats = pd.merge(lag_feats, group_feats_1, left_on=con_attr, right_index=True, how='left')
    lag_feats = pd.merge(lag_feats, group_feats_0, left_on=con_attr, right_index=True, how='left')
    lag_feats = pd.merge(lag_feats, group_feats_r, left_on=con_attr, right_index=True, how='left')


##======================================================================================================================
def get_last_buy(x):
    stop = 0
    for i in [0, 1, 2, 3, 4]:
        if x.values[i] == 1:
            stop = 5 - i
            break
    return stop


def get_first_buy(x):
    start = 0
    for i in [4, 3, 2, 1, 0]:
        if x.values[i] == 1:
            start = 5 - i
            break
    return start


def get_buy_len(x):
    x_value = x.values
    if x_value[-1] != 0:
        len1 = x_value[-1] - x_value[-2] + 1
    else:
        len1 = 0
    return len1


def add_com_features(lag_feats):
    for i in range(24):
        index_list = [11 + i, 35 + i, 59 + i, 83 + i, 107 + i]
        lag_feats['prod_sum_' + str(i)] = lag_feats.iloc[:, index_list].sum(axis=1)
        lag_feats['first_buy_' + str(i)] = lag_feats.iloc[:, index_list].apply(lambda x: get_first_buy(x), axis=1)
        lag_feats['last_buy_' + str(i)] = lag_feats.iloc[:, index_list].apply(lambda x: get_last_buy(x), axis=1)
        lag_feats['leng_buy_' + str(i)] = lag_feats.loc[:, ['first_buy_' + str(i), 'last_buy_' + str(i)]].apply(
            lambda x: get_buy_len(x), axis=1)

    pro_sum_list = ['prod_sum_' + str(i) for i in range(24)]
    pro_rank_list = ['prod_rank_' + str(i) for i in range(24)]
    lag_feats[pro_rank_list] = lag_feats[pro_sum_list].apply(lambda x: x.rank(ascending=False).astype('int'), axis=1)

    import_col = [target_cols[i] for i in [0, 2, 4, 9, 10, 11, 15, 16, 19, 20, 21]]
    for i in range(1, 6):
        pre_import_col = [str(i) + '_' + col for col in import_col]
        lag_feats[str(i) + '_11_sum_import'] = lag_feats[pre_import_col].sum(axis=1)
    for col in import_col:
        lag_feats['1_im_' + col] = lag_feats['1_' + col]

    com_col = [[0, 2], [7, 8, 9], [9, 10, 11], [19, 20, 21], [16, 19, 20, 21]]
    for x in range(4):
        import_col = [target_cols[i] for i in com_col[x]]
        for i in range(1, 6):
            pre_import_col = [str(i) + '_' + col for col in import_col]
            lag_feats[str(i) + '_' + str(x + 1) + '_s_sum_import'] = lag_feats[pre_import_col].sum(axis=1)
    return lag_feats
    # =======================================================================================================================
    columns = ['age', 'antiguedad', 'renta', 'sexo', 'ind_actividad_cliente', 'segmento', 'ind_nuevo', 'tiprel_1mes', 'indext']
    columns1 = []
    target_cols1 = [target_cols[i] for i in [0, 2, 4, 5, 6, 7, 9, 10, 11, 15, 16, 17, 19, 20, 21]]
    for i in range(1, 6):
        columns1.extend([str(i) + '_' + col for col in target_cols1])

    train_X = pd.read_csv(data_path + 'train_feats_users.csv', usecols=columns + ['label'])
    for col in ['sum', 'renta', 'canel', 'pais', 'com20', 'lagf1', 'sum8']:
        if col == 'lagf1':
            train_temp = pd.read_csv(data_path + 'train_feats_' + col + '.csv', usecols=columns1)
        else:
            train_temp = pd.read_csv(data_path + 'train_feats_' + col + '.csv')
        train_X = pd.concat([train_X, train_temp], axis=1)
    del train_temp

    test_X = pd.read_csv(data_path + 'test_feats_users.csv', usecols=columns)
    for col in ['sum', 'renta', 'canel', 'pais', 'com20', 'lagf1', 'sum8']:
        if col == 'lagf1':
            test_temp = pd.read_csv(data_path + 'test_feats_' + col + '.csv', usecols=columns1)
        else:
            test_temp = pd.read_csv(data_path + 'test_feats_' + col + '.csv')
        test_X = pd.concat([test_X, test_temp], axis=1)
    del test_temp
