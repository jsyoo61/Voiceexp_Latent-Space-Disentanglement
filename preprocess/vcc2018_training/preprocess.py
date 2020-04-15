import os
import time

from speech_tools import *

set = 'inset'
mode = 'train'

if set == 'inset':
    if mode == 'train':
        data_dir_2 = 'inset/inset_train'
    elif mode == 'dev':
        data_dir_2 = 'inset/inset_dev'
    elif mode == 'test':
        data_dir_2 = 'inset/inset_test'
    else:
        assert False, 'mode error'
elif set == 'outset':
    if mode == 'train':
        data_dir_2 = 'outset/outset_train'
    elif mode == 'dev':
        data_dir_2 = 'outset/outset_dev'
    elif mode == 'test':
        data_dir_2 = 'outset/outset_test'
    else:
        assert False, 'mode error'
else:
    assert False, 'set error'

data_dir_1 = os.path.join('../../../corpus')
data_dir = os.path.join(data_dir_1, data_dir_2)
exp_dir = os.path.join('processed')
speaker_list = os.listdir(data_dir)
start_time = time.time()

sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128

for speaker in speaker_list:

    train_A_dir = os.path.join(data_dir, speaker)
    exp_A_dir = os.path.join(exp_dir, speaker)

    os.makedirs(exp_A_dir, exist_ok=True)
    print('Loading {} Wavs...'.format(speaker))
    file_list, wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)

    print('Extracting acoustic features...')

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate,
                                                                    frame_period=frame_period, num_mcep=num_mcep)

    print('Calculating F0 statistics...')

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)

    print('Log Pitch {}'.format(speaker))
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))

    print('Normalizing data...')

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = mcs_normalization_fit_transoform(
        mcs=coded_sps_A_transposed)
    #print(np.shape(coded_sps_A_norm))
    print('Saving {} data...'.format(speaker))

    file_dir = os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep))
    save_pickle(os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)),
                (file_list, coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A))

end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')

print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
