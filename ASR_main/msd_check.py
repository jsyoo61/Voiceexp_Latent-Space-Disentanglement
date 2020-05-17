import os
import time
import librosa
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
from performance_measure import mcd_cal, msd_cal, gv_cal
from experiment import Experiment
from speech_tools import *
from tools.tools import read, save_pickle
self = Experiment(debug=True, new=False, exp_name='VAElr_expdecay1e-2_1e-4_KLDrec/VAElr_expdecay1e-2_1e-4_KLDrec_1_10')
summary = self.test(self.dirs['validation_data'], self.dirs['validation_pathlist'])
print(summary.mean())

test_pathlist = read(self.dirs['validation_pathlist']).splitlines()
sample_per_path = 10

# 1] Convert all data
print('Loading&Converting test data...')
converted_mcep_list = list()
target_mcep_list = list()
tested_pathlist = list()
start_time = time.time()
for speaker_A in self.speaker_list:
    for speaker_B in self.speaker_list:
        n_sample = 0
        if speaker_A == speaker_B:
            continue
        for test_path in test_pathlist:
            if n_sample >= sample_per_path:
                break
            # Specify conversion details
            conversion_path_sex, filename_src, filename_trg = test_path.split()
            src_speaker = filename_src.split('_')[0]
            trg_speaker = filename_trg.split('_')[0]
            ##### FOR TESTING, ERASE LATER
            # if src_speaker not in self.speaker_list or trg_speaker not in self.speaker_list:
            if src_speaker != speaker_A or trg_speaker != speaker_B:
                continue
            n_sample += 1

            wav_dir = os.path.join(self.dirs['validation_converted_best'], test_path + '.wav')
            wav, _ = librosa.load(wav_dir, sr=16000, mono=True)
            wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
            f0, timeaxis, sp, ap, coded_sp_converted = world_decompose(wav=wav, fs=16000, frame_period=5.0)

            # Define datapath
            data_B_dir = os.path.join(self.dirs['validation_data'], trg_speaker, '{}.p'.format(filename_trg))
            coded_sp_trg, _, _ = load_pickle(data_B_dir)

            converted_mcep_list.append(coded_sp_converted)
            target_mcep_list.append(coded_sp_trg)
            tested_pathlist.append(test_path)
pool = Pool()
mcd_list = pool.starmap(mcd_cal, zip(converted_mcep_list, target_mcep_list))
msd_vector_list = pool.starmap(msd_cal, zip(converted_mcep_list, target_mcep_list, itertools.repeat('vector') ))
gv_list = pool.starmap(gv_cal, zip(converted_mcep_list))
pool.close()
pool.join()

# 3] Gather Results
print('Calculation complete.')
test_result = pd.DataFrame(index = tested_pathlist, columns = self.performance_measure_index, dtype = float)
for test_path, mcd, msd_vector, gv in zip(tested_pathlist, mcd_list, msd_vector_list, gv_list):
    test_result.loc[test_path] = mcd, msd_vector, gv

save_pickle(test_result, os.path.join(self.dirs['validation'], 'msd_test.p'))
print(test_result.mean())
end_time = time.time()
time_elapsed = end_time - start_time
print('Time elapsed for testing: {:0.1f}m {:0.5}s'.format( time_elapsed // 60, time_elapsed % 60 ))
