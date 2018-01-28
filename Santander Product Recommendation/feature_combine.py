import pandas as pd
import numpy as np
import itertools

target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1',
               'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
               'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
               'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
               'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


def add_com_feats(lag_feats):
    com_feats = [target_cols[i] for i in [0, 2, 15, 16, 19, 20, 21]]
    for x, com_cols in enumerate(itertools.combinations(com_feats, 4)):
        for i in range(1, 6):
            com_col = [str(i) + '_' + col for col in com_cols]
            lag_feats[str(x) + '_com4_' + str(i)] = lag_feats[com_col].sum(axis=1)

    return lag_feats.iloc[:, -175:]


if __name__ == "__main__":
    data_path = '../input/feats/'
    train_lag5 = pd.read_csv(data_path + 'train_feats_lag5.csv')
    train_add5 = add_com_feats(train_lag5)
    train_add5.to_csv(data_path + 'train_feats_come175.csv', index=False)

    test_lag5 = pd.read_csv(data_path + 'test_feats_lag5.csv')
    test_add5 = add_com_feats(test_lag5)
    test_add5.to_csv(data_path + 'test_feats_come175.csv', index=False)
