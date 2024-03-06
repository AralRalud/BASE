# BASE: TABLE 2; on multisite dataset
# The data used for this example was randomly generated.
from os.path import join

import numpy as np
import pandas as pd


import src.config as config
from src.data.build_dataset_functions import categorize
from src.metrics import maxMAE, RMSE, R_squared, pearsons_r


if __name__ == '__main__':

    df_pred = pd.read_csv(join(config.project_root_path, 'predictions', 'predictions_single-visit_all_GENERATED.csv'))
    # create mean ensemble
    df_pred_mean = df_pred.groupby(['image_session_id', 'model_name', 'data_fold']).mean().reset_index()
    # compute error
    df_pred_mean['err'] = df_pred_mean['y_pred'] - df_pred_mean['y']
    df_pred_mean['abs_err'] = df_pred_mean['err'].abs()
    # Test set
    df_pred_mean_test = df_pred_mean[df_pred_mean.data_fold == 'test']
    df_pred_mean_test.reset_index(inplace=True, drop=True)

    #%% Evaluate bias correction on TEST set MAE
    #MAE, MAE, medAE + CI
    # MEAN MODEL
    # average error
    nME = df_pred_mean_test.groupby(by=['model_name'])['err'].mean()
    # normal MAE
    nMAE = df_pred_mean_test.groupby(by=['model_name'])['abs_err'].mean()

    #  MAE+sd ME+sd
    nMEsd = df_pred_mean_test.groupby(by=['model_name'])['err'].std()
    nMAEsd = df_pred_mean_test.groupby(by=['model_name'])['abs_err'].std()

    # max MAE
    cat_10y = np.append(np.append(np.array([18]), np.arange(25, 86, 10)), np.array([100]))
    maxMAE = df_pred_mean_test.groupby(by=['model_name'])[['y', 'y_pred']].\
        apply(maxMAE, y='y', y_pred='y_pred', categories=cat_10y)

    # RMSE
    rmse = df_pred_mean_test.groupby(by=['model_name'])[['y', 'y_pred']].\
        apply(RMSE, y='y', y_pred='y_pred')

    # R squared
    R2 = df_pred_mean_test.groupby(by=['model_name'])[['y', 'y_pred']].\
        apply(R_squared, y='y', y_pred='y_pred')

    # r
    r = df_pred_mean_test.groupby(by=['model_name'])[['y', 'y_pred']].\
        apply(pearsons_r, y='y', y_pred='y_pred')

    # concatenate
    mean_all_with_std = pd.concat((nME, nMEsd,  nMAE, nMAEsd, maxMAE, rmse, R2, r), axis=1)
    mean_all_with_std.columns = ['ME', 'ME std', 'MAE', 'MAE std', 'mMAE', 'RMSE', 'R2', 'r']
    print('DATA WAS RANDOMLY GENERATED!')
    print(mean_all_with_std.to_latex(float_format="%.2f"))