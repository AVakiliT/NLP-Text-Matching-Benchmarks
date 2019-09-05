import math
import time
from multiprocessing.pool import Pool
from joblib import Parallel, delayed


def f(x):
    return math.sqrt(x ** 2)

n = 2_000_000

now = time.time()
a = [f(x) for x in range(n)]
print(time.time() - now)

now = time.time()
with Pool(16) as pool:
    a = pool.map(f, range(n))
print(time.time() - now)

now = time.time()
a = Parallel(n_jobs=16)(delayed(f)(x) for x in range(n))
print(time.time() - now)

now = time.time()
a = Parallel(n_jobs=16, backend='multiprocessing')(delayed(f)(x) for x in range(n))
print(time.time() - now)



