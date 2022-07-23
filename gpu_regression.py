import numpy as np
import scipy
from scipy import stats
from timeit import default_timer as timer

import cupy as cp
import cupyx
# release all GPU memory:
# cp._default_memory_pool.free_all_blocks()

import warnings
warnings.filterwarnings("ignore")


# enabling unified memory (see https://youtu.be/_AKDqw6li58?t=853)
# pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
# cp.cuda.set_allocator(pool.malloc)


# free all memory in the beginning
cp._default_memory_pool.free_all_blocks()

SIZE = int(10e6);
VARS = 5;
rho = 0.2;

np.random.seed(321)


ts = timer();
X = np.ones(shape=(SIZE, VARS+1))
Y = np.pi*np.ones(shape=(SIZE, 1))
for i in range(VARS):
    X[:,(i+1):(i+2)] = rho*X[:,(i):(i+1)] + (np.sqrt(1-rho**2))*np.random.normal(size = (SIZE, 1))
    Y = Y + np.cos(i)*X[:,(i+1):(i+2)] + np.tan(i+1)*np.random.normal(size = (SIZE, 1))
te = timer();
print('X and Y creation: ', round(1000*(te-ts), 2), 'ms');


ts = timer();
d_X = cp.asarray(X);
d_Y = cp.asarray(Y);
te = timer();
print('X,Y copy to CUDA: ', round(1000*(te-ts), 2), 'ms');


def cpu_regression(X, Y, print = False):
    ts = timer();
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)
    Y_pred = np.dot(X, beta);  # model.predict(X)
    mse = (np.square(Y - Y_pred)).mean();  # mean_squared_error(Y, Y_pred)
    std_err = np.sqrt(mse * np.diag(np.linalg.inv(np.dot(X.transpose(), X))))
    tval = beta.transpose() / std_err;
    r2 = np.corrcoef(Y.transpose(), Y_pred.transpose())[0, 1] ** 2;
    te = timer();
    if print == True:
        print('\n\n', ' Coef:   ', np.round(beta.transpose(), 4), '\n',
              'std.err: ', np.round(std_err, 4), '\n',
              'tval:    ', np.round(tval, 4), '\n',
              'r2:  ', np.round(r2, 4), '\t', 'mse:  ', np.round(mse, 4), '\t', 'n_obs:  ', SIZE, '\t', 'k:  ', VARS);
    return (ts, te, beta, std_err, tval, r2, mse);


def gpu_regression(d_X, d_Y, print = False):
    ts = timer();
    d_beta = cp.dot(cp.dot(cp.linalg.inv(cp.dot(d_X.transpose(), d_X)), d_X.transpose()), d_Y)
    d_Y_pred = cp.dot(d_X, d_beta);
    d_mse = (cp.square(d_Y - d_Y_pred)).mean();
    d_std_err = cp.sqrt(d_mse * cp.diag(cp.linalg.inv(cp.dot(d_X.transpose(), d_X))))
    d_tval = d_beta.transpose() / d_std_err;
    d_r2 = cp.corrcoef(d_Y.transpose(), d_Y_pred.transpose())[0, 1] ** 2;
    te = timer();
    if print == True:
        print('\n\n', ' Coef:   ', cp.round(d_beta.transpose(), 4), '\n',
              'std.err: ', cp.round(d_std_err, 4), '\n',
              'tval:    ', cp.round(d_tval, 4), '\n',
              'r2:  ', cp.round(d_r2, 4), '\t', 'mse:  ', cp.round(d_mse, 4), '\t', 'n_obs:  ', SIZE, '\t', 'k:  ', VARS);
    return (ts, te, d_beta, d_std_err, d_tval, d_r2, d_mse);


(ts, te, beta, std_err, tval, r2, mse) = cpu_regression(X, Y)
print('CPU (np) computation: ', round(1000*(te-ts), 2), 'ms');

(ts, te, d_beta, d_std_err, d_tval, d_r2, d_mse) = gpu_regression(d_X, d_Y)
print('GPU (cp) computation: ', round(1000*(te-ts), 2), 'ms');
(ts_2, te_2, del__, del__, del__, del__, del__) = gpu_regression(d_X, d_Y)
print('GPU (cp) computation (2nd time): ', round(1000*(te_2-ts_2), 2), 'ms');


all_close = np.allclose(beta, d_beta) * np.allclose(mse, d_mse) * np.allclose(std_err, d_std_err) * np.allclose(tval, d_tval) * np.allclose(r2, d_r2)
print('all_close: ', all_close)

del X, Y, d_X, d_Y;
cp._default_memory_pool.free_all_blocks();

# print('\n\n', ' Coef:   ', np.round(beta.transpose(), 4), '\n',
#       'std.err: ', np.round(std_err, 4), '\n',
#       'tval:    ', np.round(tval, 4), '\n\n',
#       'r2:  ', np.round(r2, 4), '\t', 'mse:  ', np.round(mse, 4), '\t', 'n_obs:  ', SIZE, '\t', 'k:  ', VARS);

