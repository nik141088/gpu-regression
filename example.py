import numpy as np
from timeit import default_timer as timer
import warnings
import cupy as cp

# import cupyx
# release all GPU memory:
# cp._default_memory_pool.free_all_blocks()

from gpu_regression import cpu_regression, gpu_regression

warnings.filterwarnings("ignore")


# enabling unified memory (see https://youtu.be/_AKDqw6li58?t=853)
# pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
# cp.cuda.set_allocator(pool.malloc)


# free all memory in the beginning
cp._default_memory_pool.free_all_blocks()

SIZE = int(10e6)
VARS = 5
rho = 0.2

np.random.seed(321)


ts = timer()
X = np.ones(shape=(SIZE, VARS+1))
Y = np.pi*np.ones(shape=(SIZE, 1))
for i in range(VARS):
    X[:, i+1:i+2] = rho*X[:, i:i+1] + (np.sqrt(1-rho**2))*np.random.normal(size=(SIZE, 1))
    Y = Y + np.cos(i)*X[:, i+1:i+2] + np.tan(i+1)*np.random.normal(size=(SIZE, 1))
te = timer()
print('X and Y creation: ', round(1000*(te-ts), 2), 'ms')


ts = timer()
d_X = cp.asarray(X)
d_Y = cp.asarray(Y)
te = timer()
print('X,Y copy to CUDA: ', round(1000*(te-ts), 2), 'ms')


ts, te, beta, std_err, t_value, r2, mse = cpu_regression(X, Y)
print('CPU (np) computation: ', round(1000*(te-ts), 2), 'ms')
ts_2, te_2, _, _, _, _, _ = cpu_regression(X, Y)
print('CPU (np) computation (2nd time): ', round(1000*(te_2-ts_2), 2), 'ms')

ts, te, d_beta, d_std_err, d_t_value, d_r2, d_mse = gpu_regression(d_X, d_Y)
print('GPU (cp) computation: ', round(1000*(te-ts), 2), 'ms')
ts_2, te_2, _, _, _, _, _ = gpu_regression(d_X, d_Y)
print('GPU (cp) computation (2nd time): ', round(1000*(te_2-ts_2), 2), 'ms')


all_close = np.allclose(beta, d_beta) * np.allclose(mse, d_mse) * np.allclose(std_err, d_std_err) * np.allclose(t_value, d_t_value) * np.allclose(r2, d_r2)
print('all_close: ', all_close)

del X, Y, d_X, d_Y
cp._default_memory_pool.free_all_blocks()
