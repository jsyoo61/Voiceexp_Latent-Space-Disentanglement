import os
import numpy as np
import pickle as pk
from tools.tools import load_pickle, read, write
id_sex = load_pickle('id_sex.p')
id_sex['280'] = 'F'
data_dir_0 = '../processed_validation'

for stype in ["inset", "outset"]:
    for dtype in ["dev", "test"]:
        data_id = stype+"_"+dtype
        data_dir_1 = os.path.join(data_dir_0, data_id)
        speaker_list = os.listdir(data_dir_1)
        conversion_path = list()
        print(data_id)

        for sex_src in ['F', 'M']:
            speaker_list_src = list()
            for speaker in speaker_list:
                if id_sex[speaker[1:]] == sex_src:
                    speaker_list_src.append(speaker)

            for sex_trg in ['F', 'M']:
                speaker_list_trg = list()
                for speaker in speaker_list:
                    if id_sex[speaker[1:]] == sex_trg:
                        speaker_list_trg.append(speaker)

                for speaker_src in sorted(speaker_list_src):
                    data_dir_src = os.path.join(data_dir_1, speaker_src)
                    utterance_list_src = os.listdir(data_dir_src)

                    for speaker_trg in sorted(speaker_list_trg):
                        if speaker_src == speaker_trg:
                            continue
                        data_dir_trg = os.path.join(data_dir_1, speaker_trg)
                        utterance_list_trg = os.listdir(data_dir_trg)

                        for utterance in sorted(utterance_list_src):

                            utterance_no = utterance.split('_')[-1].split('.')[0]
                            parallel_filename = speaker_trg + '_' + utterance_no + '.p'
                            if parallel_filename in utterance_list_trg:
                                conversion_path.append(sex_src+sex_trg+' '+utterance.split('.')[0]+' '+parallel_filename.split('.')[0])

        conversion_path = '\n'.join(conversion_path)
        save_dir = data_id+'.lst'
        write(save_dir, conversion_path)
print('Process Complete.')
