import numpy as np
import matplotlib as plt 

def build_timeseries(mat, input_cols, output_col, time_steps, target_time_steps):
    # input_cols is a list of columns that would act as input columns
    # output_col is the column that would act as output column
    # total number of time-series samples = len(mat) - time_steps - target_time_steps + 1
    dim_0 = mat.shape[0] - time_steps - target_time_steps + 1
    dim_1 = len(input_cols)
    
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):  
        x[i] = mat[i: time_steps + i, input_cols]
        y[i] = mat[time_steps + target_time_steps + i - 1, output_col]

    return x, y

def convert_price(pred, sc, feature_num, out_col):
# Return predicted price for given input
    pred_reverted = np.zeros(shape=(pred.shape[0], feature_num))
    pred_reverted[:, out_col] = pred
    pred_reverted = sc.inverse_transform(pred_reverted)
    predicted_price = pred_reverted[:, out_col]
    return predicted_price