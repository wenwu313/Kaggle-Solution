import pandas as pd
import numpy as np

# test_id = pd.read_csv('../input/test_ver2.csv', usecols=['ncodpers'])
# nn_preds = pd.read_csv('../input/ensemble/taozi.csv')
# nn_preds = pd.merge(test_id, nn_preds, on = 'ncodpers', how='left')
# del nn_preds['ncodpers']
# nn_preds.to_csv('../input/ensemble/nn_preds.csv',index = False)

nn_preds = pd.read_csv('../input/ensemble/nn_preds.csv')
xgb_preds = pd.read_csv('../input/ensemble/xgb_preds.csv')

preds = (nn_preds + xgb_preds) / 2
target_cols = preds.columns
del nn_preds, xgb_preds

# target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:, :7]
test_id = np.array(pd.read_csv('../input/test_ver2.csv', usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
out_df.to_csv('../submit/sub_com.csv', index=False)
