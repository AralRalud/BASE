# BASE: TABLE 4 ; Results of reproducibility metrics on test retest dataset
# The data used for this example was randomly generated.
#
import os
from os.path import join

import numpy as np
import pandas as pd

import src.config as config
results = pd.read_csv(join(config.project_root_path,'predictions', 'predictions_test-retest_GENERATED.csv'))

# In the case of test-retest dataset, we need the mean predictions and all 5 seperate runs
# create mean ensemble
results_mean = results.groupby(['subject_id', 'image_session_id', 'model_name']).mean().reset_index()
# # compute error TODO TODO TODO
results_mean['err'] = results_mean['y_pred'] - results_mean['y']
results_mean['abs_err'] = results_mean['err'].abs()

results['err'] = results['y_pred'] - results['y']
results['abs_err'] = results['err'].abs()
#%% Difference between predicted age at visit 2 and visit 1
# add visit number
results['visit'] = results.groupby(['model_name', 'seed', 'subject_id'])['image_session_id'].transform(lambda x: x.astype('category').cat.codes + 1)
results_wide = results.pivot_table(index=['subject_id', 'model_name', 'seed', 'y'], columns='visit',
                                   values=['y_pred', 'image_session_id'], aggfunc='first').reset_index()
results_wide.columns = [f'{col}_{lvl}' if lvl else col for col, lvl in results_wide.columns.values]

results_wide['diff'] = results_wide['y_pred_1'] - results_wide['y_pred_2']
mean_diff = results_wide.groupby('model_name')['diff'].mean()
avg_std_diff = results_wide.groupby(['model_name']).std()['diff'].mean(level=0)

# average standard deviation of predicted age per scan
avg_sd_y_pred = results.groupby(['model_name', 'image_session_id', 'subject_id', 'visit'])[['y', 'y_pred']].std().\
    reset_index().groupby('model_name')['y_pred'].mean()

paper_table = pd.concat((avg_sd_y_pred, mean_diff, avg_std_diff),  axis=1)
paper_table.columns = ['avg_std_y_pred', 'mean_diff', 'std_diff']
print(paper_table.round(2))

#  ICC computed in R (03_test-rest_ICC.Rmd)
