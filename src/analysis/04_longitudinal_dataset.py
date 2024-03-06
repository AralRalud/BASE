# BASE: TABLE 5 ; Results on longitudinal dataset
# The data used for this example was GENERATED.
#
from os.path import join

import pandas as pd
import numpy as np
from itertools import combinations

import src.config as config
from src.metrics import maxMAdE


results = pd.read_csv(join(config.project_root_path, 'predictions', 'predictions_multiple-visits_GENERATED.csv'))
# create mean ensemble -- ALREADY DONE
# results_mean = results.groupby(['subject_id', 'model_name']).mean().reset_index()
results_mean = results.copy()

#%% compute predicted difference
# compute difference in deltas (y_c-y_b) - (y'_c-y'_b)
results_mean['y_delta'] = results_mean.y_2 - results_mean.y_1
results_mean['y_pred_delta'] = results_mean.y_pred_2 - results_mean.y_pred_1

results_mean['y_pred_delta_diff'] = results_mean['y_pred_delta'] - results_mean['y_delta']
results_mean['y_pred_delta_diff_abs'] = results_mean['y_pred_delta_diff'].abs()

# average delta error
nMDE = results_mean.groupby(by=['model_name'])['y_pred_delta_diff'].mean()
nMDEsd = results_mean.groupby(by=['model_name'])['y_pred_delta_diff'].std()
# MADE -- mean absolute delta error
nMADE = results_mean.groupby(by=['model_name'])['y_pred_delta_diff_abs'].mean()
nMADEsd = results_mean.groupby(by=['model_name'])['y_pred_delta_diff_abs'].std()
# maxMAE -- approx 10 year categories ([18,25), [25, 35), .... [75, 85), [85, 100))
cat_10y = np.append(np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100]))
maxMADE_10y = results_mean.groupby(by=['model_name'])[['y_1', 'y_delta', 'y_pred_delta']].\
              apply(maxMAdE, y_cat='y_1', y='y_delta', y_pred='y_pred_delta', categories=cat_10y)

paper_table2 = pd.concat(('$' + nMDE.round(2).astype(str) + '\pm' + nMDEsd.round(2).astype(str) + '$',
                             '$' + nMADE.round(2).astype(str) + '\pm' + nMADEsd.round(2).astype(str) + '$',
                             maxMADE_10y),
                         keys=['MdE', 'MAdE', 'mMAdE'],  axis=1)

print(paper_table2.to_latex(float_format="%.2f", escape=False, label="table:longitudinal-new"), )
