import numpy as np
import pandas as pd
from collections import defaultdict

pd.options.mode.chained_assignment = None

target_col = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',  'ind_cder_fin_ult1',
              'ind_cno_fin_ult1', 'ind_ctju_fin_ult1','ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
              'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1', 'ind_dela_fin_ult1',
              'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',  'ind_plan_fin_ult1',
              'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
              'ind_viv_fin_ult1', 'ind_nomina_ult1',  'ind_nom_pens_ult1', 'ind_recibo_ult1']
use_cols = ['ncodpers'] + target_col + ['sexo','renta']

def get_overbest(df_train):
    overbest_dict = {}
    for col_name in use_cols[1:25]:
        overbest_dict[col_name] = np.sum(df_train[col_name])
    top_products = sorted(overbest_dict, key = overbest_dict.get,reverse = True)
    return top_products

def get_eachbest(df_train):
    df_group = df_train[target_col].groupby(df_train['ncodpers']).sum()
    eachbest_dict = defaultdict(list)
    for ind,row in df_group.iterrows():
        row = row[row != 0].sort_values(ascending=False)
        eachbest_dict[ind] = list(row.index)
    return eachbest_dict

def get_lastinstance(last_instance_df):
    cust_dict = {}
    target_cols = np.array(use_cols[1:25])
    for ind, row in last_instance_df.iterrows():
        cust = row['ncodpers']
        used_products = set(target_cols[np.array(row[1:25] == 1)])
        cust_dict[cust] = used_products
    return cust_dict

def get_similardict(df_train):
    df_train['renta'].fillna(0, inplace=True)
    df_group1 = df_train[['renta','ind_ahor_fin_ult1']].groupby(df_train['ncodpers']).mean()
    mapping = {}
    for ind, row in df_group1.iterrows():
        if row['renta'] == 0:
            mapping[ind] = '0'
        elif row['renta'] < 45542.97:
            mapping[ind] = '1'
        elif row['renta'] < 57629.67:
            mapping[ind] = '2'
        elif row['renta'] < 68211.78:
            mapping[ind] = '3'
        elif row['renta'] < 78852.39:
            mapping[ind] = '4'
        elif row['renta'] < 90461.97:
            mapping[ind] = '5'
        elif row['renta'] < 103855.23:
            mapping[ind] = '6'
        elif row['renta'] < 120063.00:
            mapping[ind] = '7'
        elif row['renta'] < 141347.49:
            mapping[ind] = '8'
        elif row['renta'] < 173418.12:
            mapping[ind] = '9'
        elif row['renta'] < 234687.12:
            mapping[ind] = '10'
        else:
            mapping[ind] = '11'
    print mapping
    df_group2 = df_train[target_col].groupby(df_train['ncodpers']).sum()
    df_group3 = df_group2[target_col].groupby(mapping).sum()

    temp_dict = defaultdict(list)
    for ind, row in df_group3.iterrows():
        row = row[row != 0].sort_values(ascending=False)
        temp_dict[ind] = list(row.index)

    similar_dict =  defaultdict(list)
    for key in list(df_group1.index):
        similar_dict[key] = temp_dict[mapping[key]]
    return similar_dict

def get_kmeansdict(df_train):
    df_group1 = df_train[target_col].groupby(df_train['ncodpers']).sum()
    df_group1.fillna(0, inplace=True)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(df_group1.values)

    mapping = {}
    for key,value in zip(list(df_group1.index),kmeans.labels_):
        mapping[key] = value
    df_group2 = df_group1[target_col].groupby(mapping).sum()
    print mapping


    temp_dict = defaultdict(list)
    for ind, row in df_group2.iterrows():
        row = row[row != 0].sort_values(ascending=False)
        temp_dict[ind] = list(row.index)

    kmeans_dict = defaultdict(list)
    for key in list(df_group1.index):
        kmeans_dict[key] = temp_dict[mapping[key]]
    return kmeans_dict


if __name__ == "__main__":
    print("0")
    df_test = pd.read_csv('../input/test_sub_1000.csv', usecols = ['ncodpers'] + target_col)
    cust_dict = get_lastinstance(df_test)
    del df_test
    print("1")
    df_train = pd.read_csv('../input/train_sub_1000.csv', usecols = use_cols)
    top_products = get_overbest(df_train)
    print("2")
    eachbest_dict = get_eachbest(df_train)
    print("3")
    # similar_dict = get_similardict(df_train)
    similar_dict = get_kmeansdict(df_train)
    print("4")
    del df_train

    sub_id = eachbest_dict.keys()
    final_preds = []

    print("Running model")
    for ncodper, each_list in eachbest_dict.iteritems():
        used_products = cust_dict.get(ncodper,[])
        similar_product = similar_dict[ncodper]
        pred_products = []
        for product in each_list:
            if product not in used_products:
                pred_products.append(product)
                if len(pred_products) == 7:
                    break
        if len(pred_products) < 7:
            for product in similar_product:
                if (product not in used_products) and (product not in pred_products):
                    pred_products.append(product)
                    if len(pred_products) == 7:
                        break
        if len(pred_products) < 7:
            for product in top_products:
                if (product not in used_products) and (product not in pred_products):
                    pred_products.append(product)
                    if len(pred_products) == 7:
                        break

        final_preds.append(" ".join(pred_products))
    out_df = pd.DataFrame({'ncodpers':sub_id,'added_products':final_preds})

    print("Generate submission...")
    sub_92 = pd.read_csv('../input/sample_submission.csv', usecols=['ncodpers']).values[:, 0]
    submit = out_df [out_df ['ncodpers'].isin(sub_92)]
    submit.loc[:,'ncodpers'] = submit.loc[:,'ncodpers'].astype('int32')
    submit.to_csv('../input/submit1.csv', index=False)
