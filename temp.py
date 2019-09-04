import numpy as np
from matplotlib.path import Path
from joblib import Parallel, delayed
import time
import sys


## Check if one line segment contains another.

def check_paths(path):
    # global a
    # print(path, a[:10])
    res = 'no cross'
    for other_path in a:
        if other_path.contains_path(path) == 1:
            res = 'cross'
            break
    return res


if __name__ == '__main__':
    ## Create pairs of points for line segments
    a = zip(np.random.rand(5000, 2), np.random.rand(5000, 2))
    a = [Path(x) for x in a]

    b = zip(np.random.rand(300, 2), np.random.rand(300, 2))

    now = time.time()
    if len(sys.argv) >= 2:
        res = Parallel(n_jobs=int(sys.argv[1]))(delayed(check_paths)(Path(points)) for points in b)
    else:
        res = [check_paths(Path(points)) for points in b]
    print("Finished in", time.time() - now, "sec")
