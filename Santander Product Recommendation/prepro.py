import numpy as np
import pandas as pd

dtype_dict = \
    {'ncodpers': 'int32', 'age': 'str', 'antiguedad': 'str', 'renta': 'str',
     'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16',
     'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16',
     'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16',
     'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16',
     'ind_cno_fin_ult1': 'float16', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16',
     'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16',
     'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16',
     'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}

user_cols = ['ncodpers', 'fecha_dato', 'age', 'antiguedad', 'renta', 'canal_entrada', 'pais_residencia',
             'sexo', 'ind_actividad_cliente', 'segmento', 'ind_nuevo', 'tiprel_1mes', 'indext', 'indresi',
             'indfall', 'indrel', 'ind_empleado']

pro_cols = \
    ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
     'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
     'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
     'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
     'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
     'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

# use_date = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28', '2015-06-28',
#             '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
use_date = ['2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', '2016-01-28']

df_train = pd.read_csv("../input/train_ver2.csv", dtype=dtype_dict, usecols=user_cols + pro_cols)

df_train = df_train[df_train['fecha_dato'].isin(use_date)]

df_train.to_csv('../input/train_ver4.csv', index=False)
# df_test = pd.read_csv("../input/test_ver2.csv", dtype={'ncodpers':'int32'},usecols= user_cols)
#
# df_test.to_csv('../input/test_ver3.csv', index = False)
