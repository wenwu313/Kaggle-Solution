import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import itertools

pd.options.mode.chained_assignment = None

mapping_dict = {
    'sexo': {'nan': 0, 'H': 0, 'V': 1},
    'ind_actividad_cliente': {'nan': 0, '0.0': 0, '0': 0, '1.0': 1, '1': 1},
    'segmento': {'nan': 0, '01 - TOP': 1, '03 - UNIVERSITARIO': 2, '02 - PARTICULARES': 3},
    'ind_nuevo': {'nan': 0, '1.0': 1, '1': 1, '0.0': 2, '0': 2},
    'tiprel_1mes': {'nan': 0, 'P': 0, 'R': 0, 'N': 0, 'I': 1, 'A': 2},
    'indext': {'nan': 0, 'S': 0, 'N': 1},
    'indresi': {'nan': 0, 'S': 1, 'N': 2},
    'indfall': {'nan': 0, 'S': 1, 'N': 2},
    'indrel': {'nan': 1, '1': 0, '99': 1, '1.0': 0, '99.0': 1},
    'ind_empleado': {'nan': 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},
    'pais_residencia': {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17,
                        'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73,
                        'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67,
                        'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20,
                        'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90,
                        'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118,
                        'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7,
                        'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4,
                        'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95,
                        'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66,
                        'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, 'nan': 1, 'LB': 81,
                        'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37,
                        'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5,
                        'QA': 58, 'MZ': 27},
    'canal_entrada': {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12,
                      'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57,
                      'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41,
                      'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32,
                      'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54,
                      'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102,
                      'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118,
                      'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59,
                      'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81,
                      'KCI': 65,
                      'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129,
                      'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60,
                      'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117,
                      'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132,
                      'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155,
                      'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152,
                      'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87,
                      'KEU': 72, 'KES': 68, 'KEQ': 138, 'nan': 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144,
                      'KFS': 38,
                      'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42,
                      'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83,
                      'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
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
    com_col = [[0, 2], [7, 8, 9], [9, 10, 11], [19, 20, 21]]
    for x in range(4):
        import_col = [target_cols[i] for i in com_col[x]]
        for i in range(1, 6):
            pre_import_col = [str(i) + '_' + col for col in import_col]
            lag_feats[str(i) + '_' + str(x + 1) + '_s_sum_import'] = lag_feats[pre_import_col].sum(axis=1)
    return lag_feats


# def add_com_features(lag_feats):
#     lag_feats['prod_sum'] = lag_feats.apply(lambda x: np.sum(x[-120:]), axis=1)
#     for i in range(24):
#         index_list = [17+i, 41+i, 65+i, 89+i, 113+i]
#         lag_feats['prod_sum_' + str(i)] = lag_feats.iloc[:,index_list].sum(axis = 1)
#
# pro_sum_list = ['prod_sum_' + str(i) for i in range(24)]
# group_feats_r = lag_feats[pro_sum_list].groupby(lag_feats['renta' ]).agg(lambda x: round(x.sum() / x.count(), 2))
# group_feats_r.columns = ['renta_r_' + str(i) for i in range(24)]
# lag_feats = pd.merge(lag_feats, group_feats_r, left_on='renta', right_index=True, how='left')
# return lag_feats


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
    del hist_data, tmp

    # this_month = add_com_features(this_month)
    # this_month.fillna(0, inplace=True)

    this_month = pd.concat([this_month, this_month_target], axis=1)
    for idx, row in this_month.iterrows():
        for i in range(0, NUM_CLASS):
            if row[(-NUM_CLASS + i)] > 0:
                x_vars_list.append(row[:-NUM_CLASS])
                y_vars_list.append(i)
    train_X = np.array(x_vars_list)
    return train_X[:, -120:], np.array(y_vars_list)
    # return train_X, np.array(y_vars_list)


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

    del hist_file, tmp
    # test_file = add_com_features(test_file)
    # test_file.fillna(0, inplace=True)

    return test_file.values[:, -120:], test_file.columns[-120:]
    # return test_file.values, test_file.columns


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    data_path = '../input/'
    print "feature extract..."

    train_file = pd.read_csv(data_path + 'train_ver3.csv',
                             dtype={'age': 'str', 'antiguedad': 'str', 'renta': 'str'},
                             usecols=user_cols)
    print datetime.datetime.now() - start_time

    train_X, train_y = process_train_data(train_file, ['2015-06-28', '2015-05-28', '2015-04-28',
                                                       '2015-03-28', '2015-02-28', '2015-01-28'])
    # train_X = train_X[:, 2:]
    print datetime.datetime.now() - start_time

    data_date = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28']
    train_file = train_file[train_file['fecha_dato'].isin(data_date)].loc[:,
                 ['ncodpers', 'fecha_dato'] + target_raw_cols]

    test_file = pd.read_csv(data_path + 'test_ver3.csv',
                            dtype={'age': 'str', 'antiguedad': 'str', 'renta': 'str'},
                            usecols=con_cols + cat_cols)

    test_X, feats = process_test_data(test_file, train_file, data_date)
    print datetime.datetime.now() - start_time

    del train_file, test_file
    # test_X = test_X[:, 2:]
    # feats = feats[2:]
    print train_X.shape, train_y.shape, test_X.shape

    df_train = pd.DataFrame(train_X, columns=feats)
    # df_train['label'] = train_y
    df_test = pd.DataFrame(test_X, columns=feats)

    df_train.to_csv(data_path + 'feats/train_feats_lag5.csv', index=False)
    df_test.to_csv(data_path + 'feats/test_feats_lag5.csv', index=False)
    print datetime.datetime.now() - start_time
