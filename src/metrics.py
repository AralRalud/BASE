import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score, mean_squared_error

from src.auxiliary import categorize

def maxMAE(df, y, y_pred, categories):
    r"""
    Compute max MAE over given categories;
    """
    y_np = df[y].to_numpy()
    y_pred_np = df[y_pred].to_numpy()
    # categorize values of y
    cat = categorize(y_np, bin=categories, int_bin=True)

    mae_max = list()
    for category in np.unique(cat):
        y_i = y_np[cat == category]
        y_pred_i = y_pred_np[cat == category]
        mae_i = np.mean(np.abs(y_pred_i - y_i))
        # append to list
        mae_max.append(mae_i)
    return max(mae_max)

def maxMAdE(df, y_cat, y, y_pred, categories):
    r"""
    Compute max MAdE (delta error) over given categories; Subjects are categorized based on y_cat. Max error values
     is computed based on y and y_pred values,
    """
    y_cat_np = df[y_cat].to_numpy()
    y_np = df[y].to_numpy()
    y_pred_np = df[y_pred].to_numpy()
    # categorize values of y
    cat = categorize(y_cat_np, bin=categories, int_bin=True)

    mae_max = list()
    for category in np.unique(cat):
        y_i = y_np[cat == category]
        y_pred_i = y_pred_np[cat == category]
        mae_i = np.mean(np.abs(y_pred_i - y_i))
        # append to list
        mae_max.append(mae_i)
    return max(mae_max)

def pearsons_r(df, y, y_pred):
    r"""
    Compute Pearson's correlation between two columns;
    """
    y_np = df[y].to_numpy()
    y_pred_np = df[y_pred].to_numpy()
    corr_mat = np.corrcoef(y_np, y_pred_np)
    return  corr_mat[0,1] #y_true, y_pred

def RMSE(df, y, y_pred):
    r"""
    RMSE between true and predicted value;
    """
    return math.sqrt(mean_squared_error(df[y].to_numpy(), df[y_pred].to_numpy()))

def R_squared(df, y, y_pred):
    r"""
    R² between true and predicted value;
    """
    y_np = df[y].to_numpy()
    y_pred_np = df[y_pred].to_numpy()

    return r2_score(y_np, y_pred_np)
