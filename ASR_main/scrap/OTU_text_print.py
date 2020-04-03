import os
from tools import *

text_dir = '../../../project3/english/txt'
speaker_list = os.listdir(text_dir)
file_list = os.listdir(os.path.join(text_dir, speaker_list[0]))
utterance_list = list()
for file in file_list: # file: p233_116.txt
    file = file.split('_')[-1] # file: 116.txt
    utterance = file.split('.')[0] # utterance: 116
    utterance_list.append(utterance)

utterance_list.sort()
for utterance in utterance_list:

    print(utterance)
    for speaker in speaker_list:
        text_dir_1 = os.path.join(text_dir, speaker)
        file = speaker + '_' + utterance + '.txt'
        text_dir_2 = os.path.join(text_dir_1, file)
        try:
            print('%s: %s'%(speaker, read_text(text_dir_2) ) )
        except:
            print('NO UTTERANCE EXISTS FOR SPEAKER: %s'%(speaker))
