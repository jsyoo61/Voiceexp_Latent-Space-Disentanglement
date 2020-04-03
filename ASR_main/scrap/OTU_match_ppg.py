import os
from tools import load_pickle

train_data_dir = 'processed'
ppg_dir = 'processed_ppgs_train/'
speaker_list = os.listdir(train_data_dir)
speaker = speaker_list[0]
speaker

for speaker in speaker_list:
    train_data_A_dir = os.path.join(train_data_dir, speaker, 'cache{}.p'.format(36))
    ppg_A_dir = os.path.join(ppg_dir, speaker, 'ppgs_train.p')

    coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A = load_pickle(train_data_A_dir)
    ppg_A = load_pickle(ppg_A_dir) # [ppg, ppg, ...], ppg.shape == (144, n)

    n_train_data = len(coded_sps_norm_A)
    n_ppg = len(ppg_A)

    assert n_train_data == n_ppg

    for i in range(n_train_data):
        print(str(coded_sps_norm_A[i].shape), str(ppg_A[i].shape))

    coded_sps_norm_A[0]
    n_ppg[0]

    coded_sps_norm_A
