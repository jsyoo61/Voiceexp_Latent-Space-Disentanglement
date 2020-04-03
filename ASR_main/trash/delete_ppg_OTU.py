import os
from speech_tools import load_pickle, save_pickle
ppg_dir = 'processed_ppgs_train'
speaker_list = os.listdir(ppg_dir)

for speaker in speaker_list:
    print('processing: %s'%(speaker))

    ppg_dir_1 = os.path.join(ppg_dir, speaker)
    file_list = os.listdir(ppg_dir_1)
    file_list.remove('ppgs_train.p')

    for file in file_list:
        file_path = os.path.join(ppg_dir_1, file)
        os.remove(file_path)
