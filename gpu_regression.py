import numpy as np
from timeit import default_timer as timer

import cupy as cp


def cpu_regression(x, y, print_info=False):
    """
    For CPU, we use np
    """
    size = x.shape[0]
    num_vars = x.shape[1]
    ts = timer()
    beta = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x)), x.transpose()), y)
    y_pred = np.dot(x, beta)  # model.predict(x)
    mse = (np.square(y - y_pred)).mean()  # mean_squared_error(y, y_pred)
    std_err = np.sqrt(mse * np.diag(np.linalg.inv(np.dot(x.transpose(), x))))
    t_value = beta.transpose() / std_err
    r2 = np.corrcoef(y.transpose(), y_pred.transpose())[0, 1] ** 2
    te = timer()
    if print_info:
        print('\n\n', ' Coef:   ', np.round(beta.transpose(), 4), '\n',
              'std.err: ', np.round(std_err, 4), '\n',
              't_value:    ', np.round(t_value, 4), '\n',
              'r2:  ', np.round(r2, 4), '\t', 'mse:  ', np.round(mse, 4), '\t', 'n_obs:  ', size, '\t', 'k:  ', num_vars)
    return ts, te, beta, std_err, t_value, r2, mse


def gpu_regression(x, y, print_info=False):
    """
    For GPU, we use cp
    """
    size = x.shape[0]
    num_vars = x.shape[1]
    ts = timer()
    beta = cp.dot(cp.dot(cp.linalg.inv(cp.dot(x.transpose(), x)), x.transpose()), y)
    y_pred = cp.dot(x, beta)
    mse = (cp.square(y - y_pred)).mean()
    std_err = cp.sqrt(mse * cp.diag(cp.linalg.inv(cp.dot(x.transpose(), x))))
    t_value = beta.transpose() / std_err
    r2 = cp.corrcoef(y.transpose(), y_pred.transpose())[0, 1] ** 2
    te = timer()
    if print_info:
        print('\n\n', ' Coef:   ', cp.round(beta.transpose(), 4), '\n',
              'std.err: ', cp.round(std_err, 4), '\n',
              't_value:    ', cp.round(t_value, 4), '\n',
              'r2:  ', cp.round(r2, 4), '\t', 'mse:  ', cp.round(mse, 4), '\t', 'n_obs:  ', size, '\t', 'k:  ', num_vars)
    return ts, te, beta, std_err, t_value, r2, mse
