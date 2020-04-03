import os
import pickle

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

data_dir = '../../corpus'
# set_type = ['inset', 'outset']
# eval_type = ['dev', 'test']
path_list = ['inset_dev', 'outset_dev']
parallel_set = {}
id_sex = load_pickle('id_sex.p')

for p in path_list:
    # p.split: inset, test
    set_type = p.split('_')[0]
    data_dir_1 = os.path.join(data_dir, set_type)
    data_dir_2 = os.path.join(data_dir_1, p)

    speaker_list = os.listdir(data_dir_2)
    print('%s: %s'%(p, speaker_list))

    # Search corpus
    for speaker in speaker_list:
        data_dir_3 = os.path.join(data_dir_2, speaker)
        file_list = os.listdir(data_dir_3)

        for file in file_list:
            # utterance_file == 'xxx.wav'
            # speaker, utterance = file.split('_')
            utterance_file = file.split('_')[-1]

            # utterance == 'xxx'
            utterance = utterance_file.split('.')[0]

            # If utterance is found for the first time, make list
            if utterance not in parallel_set:
                parallel_set[utterance] = list()
            parallel_set[utterance].append(speaker)

# Remove single utterances
utterance_to_delete = list()
for utterance in parallel_set:
    # If only one speaker spoke the utterance, remove that utterance
    if len(parallel_set[utterance]) == 1:
        utterance_to_delete.append(utterance)

for utterance in utterance_to_delete:
    del parallel_set[utterance]

for utterance in parallel_set:
    print_list = list()
    for speaker in parallel_set[utterance]:
        # speaker: 'p235', speaker[1:]: '235'
        speaker_num = speaker[1:]

        # Sex info. some speakers don't have sex info
        try:
            sex = id_sex[speaker_num]
        except:
            sex = '?'

        print_content = speaker + '_' + sex
        print_list.append(print_content)

    print('%s: %s'%(utterance, print_list) )
    # print('%s: %s'%(utterance, parallel_set[utterance]) )
