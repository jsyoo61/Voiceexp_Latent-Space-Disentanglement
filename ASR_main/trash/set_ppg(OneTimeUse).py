import os
from speech_tools import load_pickle, save_pickle
data_dir = os.path.join('../../corpus/inset/inset_train')
speaker_list = os.listdir(data_dir)
ppg_dir = 'processed_ppgs_train'

for speaker in speaker_list:
    print('processing: %s'%(speaker))

    train_dir = os.path.join(data_dir, speaker)
    utterance_list = os.listdir(train_dir)
    ppg_dir_1 = os.path.join(ppg_dir, speaker)
    save_dir = os.path.join(ppg_dir_1, 'ppgs_train.p')

    if not os.path.isdir(ppg_dir_1):
        print('no speaker!!!')
        continue

    if os.path.exists(save_dir):
        print('already processed!')
        continue

    ppgs_list = list()
    for utterance in utterance_list:
        utterance_name = utterance.split('.')[0]
        pickle_name = utterance_name + '.pk'
        pickle_path = os.path.join(ppg_dir_1, pickle_name)
        ppgs = load_pickle(pickle_path)
        ppgs_list.append(ppgs.T)

    save_pickle(save_dir, ppgs_list)
