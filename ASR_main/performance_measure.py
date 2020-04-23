import pysptk, librosa, numpy as np
from speech_tools import *
from multiprocessing import Pool

def dtw(x, y, dist, warp = 1, w = 'inf'):
    '''Dynamic Time Warping of two sequences, accelerated with
    matrix operations instead of sequential vector operations

    Parameters
    ----------
    x: numpy.ndarray
        x.ndim == 1 or 2
    y: numpy.ndarray
        y.ndim == 1 or 2 (consistent with x)
    dist: function
        function which measures distance between frames from x and y
        if (ndim == 1)
            function should compare entry to entry(x[i], y[j]), and return single entry
        if (ndim == 2)
            function should compare vector to whole matrix(x[i], y), and return single vector
            it compares single vector frame from x to all vector frames in y.
            ex) x.shape == (T1, 10)
                y.shape == (T2, 10)
                dist(x[i], y)    ===> How dist is used
                dist(x[i], y).shape == (T2,)

    warp: maximum warp amount when finding path
        (Not Supported Yet)
    w: window size when computing cost matrix
        (Not Supported Yet)

    Returns
    -------
    distance: minimum sum of distance across all frames
    path: DTW path of x, y in list form

    Examples
    --------
    >>> from dtw import dtw
    >>> x = np.arange(10).reshape(5,2)
    >>> y = np.arange(20).reshape(10,2)
    >>> RMS = lambda x, y: np.sqrt(np.mean(np.sum((x - y) ** 2, axis = 0))) # Root Mean Square function
    >>> distance, path = dtw(x, y, dist=RMS)
    >>> distance

    >>> path[0] # path of x

    >>> path[1] # path of y

    '''
    assert x.ndim == y.ndim, 'The number of dimensions do not match\n x: %s, y: %s '%(x.ndim, y.ndim)
    assert x.ndim < 3, 'Maximum number of dimensions are 2\n, ndim: %s'%(x.dim)
    # 0] Preparation
    T1 = len(x)
    T2 = len(y)
    swap = False
    if T1 > T2:
        swap = True
        x, y = y, x
        T1, T2 = T2, T1
    # always T1 < T2, to accelerate calculation
    cost_matrix_ = np.zeros((T1+1, T2+1))
    cost_matrix_[1:, 0] = float('inf')
    cost_matrix_[0, 1:] = float('inf')
    cost_matrix = cost_matrix_[1:, 1:] # View

    # 1] Get cost matrix: dist(vector, matrix) -> Faster calculation
    # Scalar sequences
    if x.ndim == 1:
        cost_matrix[:, :] = dist(np.expand_dims(x,-1), np.broadcast_to(y, (T1,T2)))
        # cost_matrix = (x - y.expand(T1, -1).T).T
    # Vector sequences
    else:
        for i, vector in enumerate(x):
            cost_matrix[i] = dist(vector, y)
    # cost_matrix_base = cost_matrix.copy() # To save cost matrix

    # 2] Compute accumulated cost path(minimum cost matrix) & minimum cost path (T2 ~ T1+T2-1)
    # Accumulate to existing cost_matrix
    path_matrix = np.zeros((T1, T2, 2), dtype=int)
    x_i = {0:-1, 1:0, 2:-1}
    y_i = {0:-1, 1:-1, 2:0}
    for i in range(T1):
        for j in range(T2):
            search_list = [ cost_matrix_[i,j], cost_matrix_[i+1,j], cost_matrix_[i,j+1] ]
            min_hash = np.argmin(search_list)
            path_matrix[i,j] = i + x_i[min_hash] , j + y_i[min_hash]
            cost_matrix[i,j] += search_list[min_hash]

    # 3] Back-Trace minimum cost path
    i, j = T1-1, T2-1
    x_path = list()
    y_path = list()
    while (j>0 or i>0): # Since T1 < T2, i=0 occurs more. So check j first
        i, j = path_matrix[i,j]
        x_path.append(i)
        y_path.append(j)
    x_path.reverse()
    y_path.reverse()
    if swap == True:
        x_path, y_path = y_path, x_path
    return cost_matrix[-1,-1], [x_path, y_path]

def mcd_vectorized(x,y):
    '''x: vector
    y: array
    (or vice versa)
    x: array
    y: vector

    returns: vector of mcd compared between x & all vectors in array
    '''
    # return 10.0 / 2.302585092994046 * 1.4142135623730951 * np.sqrt(np.sum((x-y)**2, axis=1))
    return 10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt(np.sum((x-y)**2, axis=1))

def load_wav_extract_mcep(converted_dir, sent):
    if '.wav' in os.path.join(converted_dir, sent):
        wav, _ = librosa.load(os.path.join(converted_dir, sent), sr=22050, mono=True)
        _, _, sp, _ = world_decompose(wav=wav, fs=22050, frame_period=5.0)
        mcep = pysptk.sp2mc(sp, 36 - 1, 0.455)
        return mcep
    else:
        print('{}: not wav'.format(os.path.join(converted_dir, sent)))

def extract_ms(mcep):
    ms = logpowerspec3(mcep)
    return ms

def mcd_cal(converted_mcep, target_mcep):
    converted_mcep = converted_mcep[:,1:]
    target_mcep = target_mcep[:,1:]
    # distance, path = estimate_twf(converted_mcep, target_mcep, fast=False)
    distance, path = dtw(converted_mcep, target_mcep, dist=mcd_vectorized)
    T = len(path[0])
    mcd = distance / T
    return mcd

def msd_cal(converted_ms, target_ms, method = 'all'):
    # Mean Distance between two vectors from each feature
    T = min(len(converted_ms), len(target_ms))
    diff = converted_ms[:T] - target_ms[:T]
    if method == 'all':
        # RMS value
        return np.sqrt(np.mean(diff ** 2))
    elif method == 'vector':
        # Mean of distance for each vector
        return np.mean(np.sqrt(np.mean(diff ** 2, axis = 0)))

def gv_cal(mcep):
    gv = np.mean(np.var(mcep, axis = 0))
    return gv

def logpowerspec(fftsize, data):
    # create zero padded data
    T, dim = data.shape
    padded_data = np.zeros((fftsize, dim))
    padded_data[:T] += data

    complex_spec = np.fft.fftn(padded_data, axes=(0, 1))
    logpowerspec = 2 * np.log(np.absolute(complex_spec))  # 2가 log 안에서 제곱 역할

    return logpowerspec

def logpowerspec2(fftsize, data):
    # create zero padded data
    T, dim = data.shape
    padded_data = np.zeros((fftsize, dim))
    padded_data[:T] += data

    complex_spec = np.fft.rfft(padded_data,fftsize, axis=0)
    R, I = complex_spec.real, complex_spec.imag
    logpowerspec2 = np.log(R * R + I * I)  # 2가 log 안에서 제곱 역할

    return logpowerspec2

def logpowerspec3(data):
    complex_spec = np.fft.rfft(data, axis=0)
    R, I = complex_spec.real, complex_spec.imag
    logpowerspec2 = 10 * np.log10(R * R + I * I)  # 2가 log 안에서 제곱 역할

    return logpowerspec2

def melcd(array1, array2):
    """Calculate mel-cepstrum distortion
    Calculate mel-cepstrum distortion between the arrays.
    This function assumes the shapes of arrays are same.
    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`) or shape (`dim`)
        Arrays of original and target.
    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion
    """
    if array1.shape != array2.shape:
        raise ValueError(
            "The shapes of both arrays are different \
            : {} / {}".format(array1.shape, array2.shape))

    if array1.ndim == 2:
        # array based melcd calculation
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    elif array1.ndim == 1:
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
    else:
        raise ValueError("Dimension mismatch")

    return mcd

def estimate_twf(orgdata, tardata, distance='melcd', fast=True, otflag=None):
    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y)
    else:
        raise ValueError('other distance metrics than melcd does not support.')

    if otflag is None:
        # use dtw or fastdtw
        if fast:
            distance, path = fastdtw(orgdata, tardata, dist=distance_func)
            twf = np.array(path).T
        else:
            distance, _, _, twf = dtw(orgdata, tardata, distance_func)

    return distance, twf

# coded_sp=load_pickle('processed/p225/cache36.p')[1]
# x=coded_sp[0].T
# y=coded_sp[9].T
# x.shape
# y.shape
# dist = lambda x, y: (x-y)**2
# dist1 = lambda x, y:  10.0 / np.log(10) * np.sqrt(2.0 * np.sum((x-y )** 2))
# dist2 = lambda x, y: 10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt( np.sum((x-y)**2, axis=1))
# from dtw import dtw as dtw_1
# result1 = dtw_1(x, y, dist=dist1)
# result2 = dtw(x, y, dist=dist2)
# print(result1[0],result2[0])
# # %%
# # result1
# # result2
# st = time.time()
# result1 = dtw.dtw(x,y, dist=dist)
# et = time.time()
# print(et - st)
# st = time.time()
# result2 = fastdtw.dtw(x,y, dist=dist)
# et = time.time()
# print(et - st)
# # %%
# st = time.time()
# result3 = torch_dtw(x,y, dist=dist)
# et = time.time()
# print(et - st)
#
# # %%
# for i in range(5):
#     st = time.time()
#     result3 = torch_dtw(y,x, dist=dist)
#     et = time.time()
#     print(et - st)
# print('--------')
# for i in range(5):
#     st = time.time()
#     result3 = torch_dtw(x,y, dist=dist)
#     et = time.time()
#     print(et - st)

# import matplotlib.pyplot as plt
# x1 = np.arange(0,5,10/200)
# x2 = np.arange(0,40,10/200)
# x3 = np.arange(0,20,10/200)
#
# T1 = x1[-1] - x1[0]
# T2 = x2[-1] - x2[0]
# T3 = x3[-1] - x3[0]
# N1=len(x1)
# N2=len(x2)
# N3=len(x3)
# y1 = np.sin(2*np.pi/T1 * x1) + np.sin(2*np.pi/T1 * N1/4 * x1)
# y1 = np.concatenate([np.sin(2*np.pi/T1 * x1), np.sin(2*np.pi/T1 * (N1)/4 * x1)])
# # y1 = np.sin(2*np.pi/T1 * x1)
# y2 = np.sin(2*np.pi/T2 * x2) + np.sin(2*np.pi/T2 * N2/4 * x2)
# y2 = np.concatenate([np.sin(2*np.pi/T2 * x2), np.sin(2*np.pi/T2 * N2/4 * x2)])
# # y2 = np.sin(2*np.pi/T2 * x2)
# y3 = np.sin(2*np.pi/T3 * N3*1/8 * x3) + np.sin(2*np.pi/T3 * N3*3/8 * x3)
# y1 = np.sin(2*np.pi/T1 * x1)
# y2 = np.sin(2*np.pi/T2 * x2)
# N1 = len(y1)
# N2 = len(y2)
# N3 = len(y3)
# plt.plot(y1)
# plt.plot(y2)
# plt.plot(y3)
# plt.plot( np.sin(2*np.pi/T1 * (N1)/ * x1))
# N1
#
# fftsize=2048
# padded_data1 = np.zeros(fftsize)
# padded_data1[:N1] = y1
# padded_data2 = np.zeros(fftsize)
# padded_data2[:N2] = y2
# padded_data3 = np.zeros(fftsize)
# padded_data3[:N3] = y3
# plt.plot(padded_data1)
# plt.plot(padded_data2)
# plt.plot(padded_data3)
# padded_data4 = np.sin(2*np.pi/fftsize * 2 * np.arange(fftsize))
# padded_data4[fftsize//2:]=0
# plt.plot(padded_data4)
#
# padded_data.shape
# complex1 = np.fft.rfft(padded_data1, fftsize)
# complex2 = np.fft.rfft(padded_data2, fftsize)
# complex3 = np.fft.rfft(padded_data3, fftsize)
# complex4 = np.fft.rfft(padded_data4, fftsize)
# plt.stem(abs(complex1))
# plt.stem(abs(complex2))
# plt.stem(abs(complex3))
# plt.stem(abs(complex4))
# len(complex1)
# len(complex2)
# abs(complex1)
# abs(complex4)
# complex_n1 = np.fft.rfft(y1)
# complex_n2 = np.fft.rfft(y2)
# plt.stem(abs(complex_n1))
# plt.stem(abs(complex_n2))
#
# ms1_ = logpowerspec2(fftsize, np.expand_dims(y1, -1))
# plt.stem(ms1_.squeeze(-1))
# plt.clf()
# ms1_.shape
# ms1.shape
# np.where((ms1!=ms1_.squeeze(-1)))
# ms1_.squeeze().shape

# c = complex1[np.where((abs(complex1)**2 != complex1.real **2 + complex1.imag**2))]
# c1=c[0]
# abs(c1)
# np.sqrt(c1.real**2+c1.imag**2)
#
# abs(c1)**2
# c1.real**2 + c1.imag**2
# c1=complex1[0]
#
# help(np.fft.rfft)
# 4096/2
# T1
# T2
#
# ms1 = np.log(abs(complex1)**2)
# ms2 = np.log(abs(complex2)**2)
# ms3 = np.log(abs(complex3)**2)
# ms4 = np.log(abs(complex4)**2)
# plt.stem(ms1)
# plt.stem(ms2)
# plt.stem(ms3)
# plt.stem(ms4)
# np.sqrt(np.mean( (ms1-ms4)**2 ))
# plt.stem(ms1-ms2)
# plt.stem(ms1-ms3)
# np.sqrt(np.mean( (ms1-ms2)**2 ) )
# np.sqrt(np.mean( (ms1-ms3)**2 ))
# np.sqrt(np.mean( (ms2-ms3)**2 ))
# np.linalg.norm(ms1-ms2)
# np.linalg.norm((ms1-ms3))
#
# np.sqrt(np.mean( (ms1_ - ms2)**2) )
# ms1_ = ms1_.squeeze()
# ms1
#
# plt.plot(ms1-ms1_)
# np.linalg.norm(ms1-ms1_)
#
# np.sqrt(np.mean( (ms1_-ms3)**2))
# np.sqrt(np.mean( (ms1_-ms2)**2))



# x=np.arange(20).reshape(10,2)
# np.linalg.norm(x, axis=0)
# y1 = np.cos(2*np.pi/10 *100* x)
# plt.plot(y1)
# y=np.cos(2*np.pi/10* 20 * x) + np.cos(2*np.pi/10* 30 * x) + np.sin(2*np.pi/10* 50 * x)
# import matplotlib.pyplot as plt
# plt.plot(x,y)
# help(np.fft.helper)
# help(np.fft.fftn)
# help(np.fft.fft)
# help(np.fft.fft2)
# help(np.fft.rfft)
# re=np.fft.rfft(y)
# res = np.fft.rfft(y1)
# plt.plot(abs(res))
# re.shape
# plt.plot(abs(re))
# plt.plot(re.real)
# plt.plot(re.imag)
# abs(re) == np.sqrt(re.imag **2 + re.real**2)
