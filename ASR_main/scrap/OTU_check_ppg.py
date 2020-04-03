import os
from speech_tools import load_pickle
ppg_dir = 'processed_ppgs_train/'

speaker_list = sorted(os.listdir(ppg_dir))
for speaker in speaker_list:
    # print(speaker)
    ppg_dir_1 = os.path.join(ppg_dir, speaker) # processed_ppgs_train/p225
    file_list = sorted(os.listdir(ppg_dir_1))
    for file in file_list:
        ppg_dir_2 = os.path.join(ppg_dir_1, file)
        try:
            ppg = load_pickle(ppg_dir_2)
        except:
            print(os.path.join(speaker, file))
