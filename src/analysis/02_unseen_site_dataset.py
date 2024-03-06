# BASE: TABLE 3; Results on unseen site dataset with the same preprocessing and new preprocessing
# The data used for this example was randomly generated.
#
from os.path import join

import pandas as pd
import numpy as np

import src.config as config
from src.metrics import maxMAE

# %% UNSEEN SITE ################################
results = pd.read_csv(join(config.project_root_path, 'predictions', 'predictions_unseen_site_all_GENERATED.csv'))
# create mean ensemble
results_mean = results.groupby(['image_session_id', 'model_name']).mean().reset_index()
# compute error
results_mean['err'] = results_mean['y_pred'] - results_mean['y']
results_mean['abs_err'] = results_mean['err'].abs()

# TABLE 3 (TOP LEFT) -- SAME preprocessing
# average error
nME = results_mean.groupby(by=['model_name'])['err'].mean()
nMEsd = results_mean.groupby(by=['model_name'])['err'].std()
# MAE
nMAE = results_mean.groupby(by=['model_name'])['abs_err'].mean()
nMAEsd = results_mean.groupby(by=['model_name'])['abs_err'].std()

# maxMAE -- approx 10 year categories ([18,25), [25, 35), .... [75, 85), [85, 100))
cat_10y = np.append(np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100]))
maxMAE_10y = results_mean.groupby(by=['model_name'])[['y', 'y_pred']]. \
  apply(maxMAE, y='y', y_pred='y_pred', categories=cat_10y)


paper_table = pd.concat(('$' + nME.round(2).astype(str) + '\pm' + nMEsd.round(2).astype(str) + '$',
                        '$' + nMAE.round(2).astype(str) + '\pm' + nMAEsd.round(2).astype(str) + '$',
                        maxMAE_10y), keys=['ME (sd)', 'MAE (sd)', 'mMAdE'],  axis=1)

print(paper_table.to_latex(float_format="%.2f", escape=False, label="table:ukb_same_preproc"), )


#TABLE 3 (BOTTOM LEFT) -- SAME preprocessing (offset corrected)
# Offset correction (Already computed in the input csv; left for completeness)
offset = results_mean.groupby(by=['model_name'])[['err', 'y']].mean()
offset.reset_index(inplace=True)


for model in offset.model_name.unique():
    offset_model = offset.loc[offset.model_name == model, 'err'].values[0]
    results_mean.loc[results_mean.model_name == model, 'y_pred_offset'] = (
            results_mean.loc[results_mean.model_name == model, 'y_pred'] - offset_model)

results_mean['err_offset'] = results_mean['y_pred_offset'] - results_mean['y']
results_mean['abs_err_offset'] = results_mean.err_offset.abs()

# average error
nME = results_mean.groupby(by=['model_name'])['err_offset'].mean()
nMEsd = results_mean.groupby(by=['model_name'])['err_offset'].std()
# MAE
nMAE = results_mean.groupby(by=['model_name'])['abs_err_offset'].mean()
nMAEsd = results_mean.groupby(by=['model_name'])['abs_err_offset'].std()

# maxMAE -- approx 10 year categories ([18,25), [25, 35), .... [75, 85), [85, 100))
maxMAE_10y = results_mean.groupby(by=['model_name'])[['y', 'y_pred_offset']].\
  apply(maxMAE, y='y', y_pred='y_pred_offset', categories=cat_10y)

paper_table2 = pd.concat(('$' + nME.round(2).astype(str) + '\pm' + nMEsd.round(2).astype(str) + '$',
                          '$' + nMAE.round(2).astype(str) + '\pm' + nMAEsd.round(2).astype(str) + '$',
                          maxMAE_10y),
                          keys=['ME (sd)', 'MAE (sd)', 'mMAdE'],  axis=1)

print(paper_table2.to_latex(float_format="%.2f", escape=False, label="table:ukb_same_preproc_offset_corrected"), )

#%% NEW PREPROCESSING ################################
# REPEAT THE SAME FOR NEW PREPROCESSING