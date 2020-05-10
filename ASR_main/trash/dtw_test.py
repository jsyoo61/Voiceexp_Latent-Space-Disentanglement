import dtw as dtw2
import fastdtw
import time
import numpy as np
import torch
from torch.multiprocessing import Pool as Pool_torch
from multiprocessing import Pool
from tools.tools import load_pickle
from performance_measure import dtw, dtw_torch

n=10
coded_sp=load_pickle('processed/p225/cache36.p')[1]
x=coded_sp[1].T
y=coded_sp[4].T

def dist_torch(x, y):
    return 10.0 / np.log(10) * np.sqrt(2.0) * torch.sqrt(torch.sum((x-y)**2, axis=1))
def dist_np(x, y):
    return 10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt(np.sum((x-y)**2, axis=1))
def dist_entry(x,y):
    return 10.0 / np.log(10) * np.sqrt(2.0 * np.sum((x-y )** 2))
# dist=lambda x,y:10.0 / np.log(10) * np.sqrt(2.0) * torch.sqrt(torch.sum((x-y)**2, axis=1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st = time.time()
pool = Pool_torch(30)
result1 = pool.starmap(dtw_torch, zip(coded_sp[:n],coded_sp[n:2*n], [dist_torch]*n, [device]*n))
pool.close()
pool.join()
# result4 = dtw_torch(x,y, dist=lambda x,y:10.0 / np.log(10) * np.sqrt(2.0) * torch.sqrt(torch.sum((x-y)**2, axis=1)), device=device)
et = time.time()
print(et - st)

# dist = lambda x, y:  10.0 / np.log(10) * np.sqrt(2.0 * np.sum((x-y )** 2))
st = time.time()
pool = Pool(30)
result2 = pool.starmap(dtw2.dtw, zip(coded_sp[:n],coded_sp[n:2*n], [dist_entry]*n))
# result1 = dtw2.dtw(x,y, dist=dist)
pool.close()
pool.join()
et = time.time()
print(et - st)

st = time.time()
pool = Pool(30)
result3 = pool.starmap(fastdtw.dtw, zip(coded_sp[:n],coded_sp[n:2*n], [dist_entry]*n))
pool.close()
pool.join()
# result2 = fastdtw.dtw(x,y, dist=dist)
et = time.time()
print(et - st)

# dist=lambda x,y:10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt(np.sum((x-y)**2, axis=1))
st = time.time()
pool = Pool(30)
result4 = pool.starmap(dtw, zip(coded_sp[:n],coded_sp[n:2*n], [dist_np]*n))
pool.close()
pool.join()
# result3 = dtw(x,y, dist=lambda x,y:10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt(np.sum((x-y)**2, axis=1)))
et = time.time()
print(et - st)
