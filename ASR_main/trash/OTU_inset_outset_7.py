'''
Trying to find utterances
that appear in each set exclusively.

Utterances that does not overlap
with any other sets
'''
import os
import pickle
from tools import *

data_dir = '../../corpus'
set_type = ['inset', 'outset']
eval_type = ['dev', 'test']
id_sex = load_pickle('id_sex.p')
set_source = dict()

# Search the source of Utterances
for s in set_type:
    data_dir_1 = os.path.join(data_dir, s) # data_dir_1: '../../corpus/inset'

    for e in eval_type:
        path = s + '_' + e
        data_dir_2 = os.path.join(data_dir_1, path) # data_dir_2: '../../corpus/inset/inset_dev'
        speaker_list = os.listdir(data_dir_2)

        # Search corpus
        for speaker in speaker_list:
            data_dir_3 = os.path.join(data_dir_2, speaker)
            file_list = os.listdir(data_dir_3)

            for file in file_list:
                # speaker, utterance = file.split('_')
                utterance_file = file.split('_')[-1] # utterance_file == 'xxx.wav'
                utterance = utterance_file.split('.')[0] # utterance == 'xxx'

                # If utterance is found for the first time, make list
                if utterance not in set_source:
                    set_source[utterance] = list()

                # append source_set to the current utterance
                if path not in set_source[utterance]:
                    set_source[utterance].append(path)

# for utterance in sorted(set_source.keys()):
#     print('%s: %s'%(utterance, set_source[utterance]))

# Remove single-source utterances
utterance_to_delete = list()
for utterance in set_source:

    # If the utterance came from more than one set
    # Delete the utterance,
    # Since we only need set-exclusive utterances
    num_set_source = len(set_source[utterance])
    if num_set_source < 4 :
        utterance_to_delete.append(utterance)

for utterance in utterance_to_delete:
    del set_source[utterance]

text_dir = '../../../project3/english/txt'
utterance_list = sorted(set_source.keys())
utterance_to_save = list()
for utterance in utterance_list:
    if utterance < '025':
        utterance_to_save.append(utterance)
utterance_list = utterance_to_save

# Print result, utterance by utterance
for utterance in utterance_list:

    text = None
    for path in set_source[utterance]: # path: inset_test

        s = path.split('_')[0] # s: inset
        data_dir_1 = os.path.join(data_dir, s)
        data_dir_2 = os.path.join(data_dir_1, path)
        speaker_list = os.listdir(data_dir_2)

        print_list = list()
        print_list_F = list()
        print_list_M = list()
        print_list_unknown = list()

        # Search corpus
        for speaker in speaker_list:
            data_dir_3 = os.path.join(data_dir_2, speaker)
            file_list = os.listdir(data_dir_3)

            file = speaker + '_' + utterance + '.wav' # file: p288_028.wav

            # If the utterance is spoken by the speaker, Append "speaker_sex"(p288_F)
            if file in file_list:

                speaker_num = speaker[1:] # speaker: 'p235', speaker[1:]: '235'

                # Sex info. some speakers don't have sex info
                try:
                    sex = id_sex[speaker_num]
                except:
                    sex = '?'

                print_content = speaker + '_' + sex # print_content: p288_F
                if sex == 'F':
                    print_list_F.append(print_content)
                elif sex == 'M':
                    print_list_M.append(print_content)
                else:
                    print_list_unknown.append(print_content)
                # print_list.append(print_content)

                # Read transcript only once
                if text == None:
                    # Read transcription
                    text_dir_1 = os.path.join(text_dir, speaker)
                    text_file = speaker + '_' + utterance + '.txt' # text_file: p_288_028.txt
                    text_dir_2 = os.path.join(text_dir_1, text_file)

                    text = read_text(text_dir_2)
                    num_words = len(text.split())
                    text = text.splitlines()[0]
                    print('%s, Word_count:%s, Transcript:%s'%(utterance, num_words, text))

        print_list_F.sort(reverse = True)
        print_list_M.sort(reverse = True)
        print_list_unknown.sort(reverse = True)
        print_list = print_list_F + print_list_M
        print('%s: %s'%(path, print_list))

    # \n
    print()


# for i in range(2):
#     print('')
#     print('b')
# # a ='asdf asf'
# a.split('\n')
# A = {3:30, 2: 20, 1:10}
# sorted(A.keys())
# txt = 'a  \n ' + ''
# txt.split()
#
# t = read_text('p288_408.txt')
# t
# t.split()
# t.split('\n')
# t.splitlines()
# t.split().join(' ')
# print('aa\rb')
# '\n'
B = [3,2,1,5,6]
B.sort()
B
