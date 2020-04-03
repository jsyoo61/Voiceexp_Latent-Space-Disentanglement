'''
Trying to find utterances
that appear in each set exclusively.

Utterances that does not overlap
with any other sets
'''
import os
import pickle

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

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

for utterance in set_source:
    print('%s: %s'%(utterance, set_source[utterance]))

print('*'*30)
# Remove multi-source utterances
utterance_to_delete = list()
for utterance in set_source:
    # print('%s: %s'%(utterance, set_source[utterance]))

    # If the utterance came from more than one set
    # Delete the utterance,
    # Since we only need set-exclusive utterances
    num_set_source = len(set_source[utterance])
    if num_set_source > 1 :
        utterance_to_delete.append(utterance)

for utterance in utterance_to_delete:
    del set_source[utterance]

# print('*'*20)
# print('Multi-source utterance deleted')

# Print utterance - source
for utterance in set_source:
    print('%s: %s'%(utterance, set_source[utterance]))

print('')

# Print result, utterance by utterance
for utterance in set_source:
    # path: inset_test
    path = set_source[utterance][0]
    # s: inset
    s = path.split('_')[0]
    data_dir_1 = os.path.join(data_dir, s)
    data_dir_2 = os.path.join(data_dir_1, path)
    speaker_list = os.listdir(data_dir_2)

    print_list = list()

    # Search corpus
    for speaker in speaker_list:
        data_dir_3 = os.path.join(data_dir_2, speaker)
        file_list = os.listdir(data_dir_3)

        file = speaker + '_' + utterance + '.wav'

        # If the utterance is spoken by the speaker
        if file in file_list:

            # speaker: 'p235', speaker[1:]: '235'
            speaker_num = speaker[1:]

            # Sex info. some speakers don't have sex info
            try:
                sex = id_sex[speaker_num]
            except:
                sex = '?'

            print_content = speaker + '_' + sex
            print_list.append(print_content)

    print('set: %s'%(path))
    print('%s: %s\n'%(utterance, print_list))



        # # Remove single utterances
        # utterance_to_delete = list()
        # for utterance in parallel_set:
        #     # If only one speaker spoke the utterance, remove that utterance
        #     if len(parallel_set[utterance]) == 1:
        #         utterance_to_delete.append(utterance)
        #
        # for utterance in utterance_to_delete:
        #     del parallel_set[utterance]
        #
        # # Print
        # print(s+'_'+e)
        # for utterance in parallel_set:
        #     print_list = list()
        #     for speaker in parallel_set[utterance]:
        #         # speaker: 'p235', speaker[1:]: '235'
        #         speaker_num = speaker[1:]
        #         sex = id_sex[speaker_num]
        #
        #         print_content = speaker + '_' + sex
        #         print_list.append(print_content)
        #
        #     print('%s: %s'%(utterance, print_content) )
