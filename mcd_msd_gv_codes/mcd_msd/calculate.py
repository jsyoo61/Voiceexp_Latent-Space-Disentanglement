def calculate_mcd_msd(alpha, fftlen, pool, sr=22050, num_mcep=36, frame_period=5.0, validation_dir='./result/output',
                      gt_dir='./VCC2018/vcc2018_evaluation'):
    converted_dirs = os.listdir(validation_dir)
    total_mcd_list = list()
    total_sample_mcd_list = list()
    f2f_mcd_list = list()
    f2m_mcd_list = list()
    m2f_mcd_list = list()
    m2m_mcd_list = list()
    sample_f2f_mcd_list = list()
    sample_f2m_mcd_list = list()
    sample_m2f_mcd_list = list()
    sample_m2m_mcd_list = list()
    total_msd_list = list()
    total_sample_msd_list = list()
    f2f_msd_list = list()
    f2m_msd_list = list()
    m2f_msd_list = list()
    m2m_msd_list = list()
    sample_f2f_msd_list = list()
    sample_f2m_msd_list = list()
    sample_m2f_msd_list = list()
    sample_m2m_msd_list = list()

    for converted_dir in converted_dirs:
        src, tar = converted_dir.split('_to_')
        converted_dir = os.path.join(validation_dir, converted_dir)
        target_dir = os.path.join(gt_dir, tar)
        sent_list = os.listdir(converted_dir)

        p = Pool(pool)
        converted_mcep_list = p.starmap(load_wav_extract_mcep, zip([converted_dir] * len(sent_list), sent_list))
        target_mcep_list = p.starmap(load_wav_extract_mcep, zip([target_dir] * len(sent_list), sent_list))
        p.close()
        p.join()

        p = Pool(pool)
        converted_ms_list = p.map(extract_ms, converted_mcep_list)
        target_ms_list = p.map(extract_ms, target_mcep_list)
        p.close()
        p.join()

        p = Pool(pool)
        mcd_list = p.starmap(mcd_cal, zip(converted_mcep_list, target_mcep_list))
        msd_list = p.starmap(msd_cal, zip(converted_ms_list, target_ms_list))
        p.close()
        p.join()

        for mcd, msd, sent in zip(mcd_list, msd_list, sent_list):
            total_mcd_list.append(mcd)
            total_msd_list.append(msd)
            if 'F' in src and 'F' in tar:
                f2f_mcd_list.append(mcd)
                f2f_msd_list.append(msd)
                if src == 'VCC2SF1' and tar == 'VCC2SF2' and sent in ['30011.wav', '30012.wav', '30013.wav']:
                    sample_f2f_mcd_list.append(mcd)
                    sample_f2f_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)

            elif 'F' in src and 'M' in tar:
                f2m_mcd_list.append(mcd)
                f2m_msd_list.append(msd)
                if src == 'VCC2SF2' and tar == 'VCC2SM2' and sent in ['30005.wav', '30009.wav', '30011.wav']:
                    sample_f2m_mcd_list.append(mcd)
                    sample_f2m_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)
            elif 'M' in src and 'F' in tar:
                m2f_mcd_list.append(mcd)
                m2f_msd_list.append(msd)
                if src == 'VCC2SM1' and tar == 'VCC2SF1' and sent in ['30004.wav', '30005.wav', '30016.wav']:
                    sample_m2f_mcd_list.append(mcd)
                    sample_m2f_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)
            elif 'M' in src and 'M' in tar:
                m2m_mcd_list.append(mcd)
                m2m_msd_list.append(msd)
                if src == 'VCC2SM2' and tar == 'VCC2SM1' and sent in ['30005.wav', '30009.wav', '30012.wav']:
                    sample_m2m_mcd_list.append(mcd)
                    sample_m2m_msd_list.append(msd)
                    total_sample_mcd_list.append(mcd)
                    total_sample_msd_list.append(msd)

        print('SRC: ', src, ' TAR: ', tar, ' MCD_MEAN: ', np.mean(mcd_list), ' MCD_STD: ', np.std(mcd_list))
        print('SRC: ', src, ' TAR: ', tar, ' MSD_MEAN: ', np.mean(msd_list), ' MSD_STD: ', np.std(msd_list))

    print()
    print('-' * 30)
    print('total ', ' MCD_MEAN: ', np.mean(total_mcd_list), ' MCD_STD: ', np.std(total_mcd_list))
    print('sample ', ' MCD_MEAN: ', np.mean(total_sample_mcd_list), ' MCD_STD: ', np.std(total_sample_mcd_list))
    print('f2f ', ' MCD_MEAN: ', np.mean(f2f_mcd_list), ' MCD_STD: ', np.std(f2f_mcd_list))
    print('f2m ', ' MCD_MEAN: ', np.mean(f2m_mcd_list), ' MCD_STD: ', np.std(f2m_mcd_list))
    print('m2f ', ' MCD_MEAN: ', np.mean(m2f_mcd_list), ' MCD_STD: ', np.std(m2f_mcd_list))
    print('m2m ', ' MCD_MEAN: ', np.mean(m2m_mcd_list), ' MCD_STD: ', np.std(m2m_mcd_list))
    print('sample_f2f ', ' MCD_MEAN: ', np.mean(sample_f2f_mcd_list), ' MCD_STD: ', np.std(sample_f2f_mcd_list))
    print('sample_f2m ', ' MCD_MEAN: ', np.mean(sample_f2m_mcd_list), ' MCD_STD: ', np.std(sample_f2m_mcd_list))
    print('sample_m2f ', ' MCD_MEAN: ', np.mean(sample_m2f_mcd_list), ' MCD_STD: ', np.std(sample_m2f_mcd_list))
    print('sample_m2m ', ' MCD_MEAN: ', np.mean(sample_m2m_mcd_list), ' MCD_STD: ', np.std(sample_m2m_mcd_list))

    print()
    print('-' * 30)
    print('total ', ' MSD_MEAN: ', np.mean(total_msd_list), ' MSD_STD: ', np.std(total_msd_list))
    print('sample ', ' MSD_MEAN: ', np.mean(total_sample_msd_list), ' MSD_STD: ', np.std(total_sample_msd_list))
    print('f2f ', ' MSD_MEAN: ', np.mean(f2f_msd_list), ' MSD_STD: ', np.std(f2f_msd_list))
    print('f2m ', ' MSD_MEAN: ', np.mean(f2m_msd_list), ' MSD_STD: ', np.std(f2m_msd_list))
    print('m2f ', ' MSD_MEAN: ', np.mean(m2f_msd_list), ' MSD_STD: ', np.std(m2f_msd_list))
    print('m2m ', ' MSD_MEAN: ', np.mean(m2m_msd_list), ' MSD_STD: ', np.std(m2m_msd_list))
    print('sample_f2f ', ' MSD_MEAN: ', np.mean(sample_f2f_msd_list), ' MSD_STD: ', np.std(sample_f2f_msd_list))
    print('sample_f2m ', ' MSD_MEAN: ', np.mean(sample_f2m_msd_list), ' MSD_STD: ', np.std(sample_f2m_msd_list))
    print('sample_m2f ', ' MSD_MEAN: ', np.mean(sample_m2f_msd_list), ' MSD_STD: ', np.std(sample_m2f_msd_list))
    print('sample_m2m ', ' MSD_MEAN: ', np.mean(sample_m2m_msd_list), ' MSD_STD: ', np.std(sample_m2m_msd_list))


def load_wav_extract_mcep(converted_dir, sent):
    if '.wav' in os.path.join(converted_dir, sent):
        wav, _ = librosa.load(os.path.join(converted_dir, sent), sr=22050, mono=True)
        _, _, sp, _ = world_decompose(wav=wav, fs=22050, frame_period=5.0)
        mcep = pysptk.sp2mc(sp, 36 - 1, 0.455)
        return mcep
    else:
        print('{}: not wav'.format(os.path.join(converted_dir, sent)))


def extract_ms(mcep):
    ms = logpowerspec2(4096, mcep)
    return ms


def mcd_cal(converted_mcep, target_mcep):
    converted_mcep = converted_mcep[:,1:]
    target_mcep = target_mcep[:,1:]
    twf = estimate_twf(converted_mcep, target_mcep, fast=False)
    converted_mcep_mod = converted_mcep[twf[0]]
    target_mcep_mod = target_mcep[twf[1]]
    mcd = melcd(converted_mcep_mod, target_mcep_mod)
    return mcd


def msd_cal(converted_ms, target_ms):
    msd = np.sqrt(np.mean(np.power((converted_ms - target_ms), 2)))
    return msd


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


import pysptk, librosa, numpy as np
from preprocess import *
from multiprocessing import Pool
from dtw import dtw
from fastdtw import fastdtw
import argparse

alpha = 0.455
fftlen = 4096

parser = argparse.ArgumentParser(description='T')

model_default = '/nas/harden.lsd/CCGAN-VC'
parser.add_argument('--model', type=str, help='model',
                        default=model_default)
parser.add_argument('--gt_dir', type=str, help='model',
                        default='./VCC2018/vcc2018_evaluation')
parser.add_argument('--pool', type=int, help='pool',
                        default=35)

argv = parser.parse_args()
gt_dir = argv.gt_dir
model = argv.model
pool = argv.pool

calculate_mcd_msd(alpha, fftlen, pool, sr=22050, num_mcep=36, frame_period=5.0, validation_dir=model,
                      gt_dir=gt_dir)
