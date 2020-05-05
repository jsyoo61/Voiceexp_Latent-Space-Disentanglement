import os
import shutil
from tools.tools import read

test_pathlist_dir = 'filelist/inset_test.lst'
test_pathlist = read(test_pathlist_dir).splitlines()
save_dir = 'reference'
os.makedirs(save_dir, exist_ok=True)

data_dir = '../../corpus/inset/inset_test'
sample_per_path=10
speaker_list = ['p225','p226','p227','p228']
for speaker_A in speaker_list:
        for speaker_B in speaker_list:
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
                source_wav_dir = os.path.join(data_dir, src_speaker, '{}.wav'.format(filename_src))
                target_wav_dir = os.path.join(data_dir, trg_speaker, '{}.wav'.format(filename_trg))
                shutil.copy(source_wav_dir, os.path.join(save_dir, '{}.wav'.format(filename_src)))
                shutil.copy(target_wav_dir, os.path.join(save_dir, '{}.wav'.format(filename_trg)))
