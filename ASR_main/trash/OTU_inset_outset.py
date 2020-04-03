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
# eval_type = ['dev', 'test']
eval_type = ['dev']
id_sex = load_pickle('id_sex.p')

for s in set_type:
    data_dir_1 = os.path.join(data_dir, s)

    for e in eval_type:
        path = s + '_' + e
        data_dir_2 = os.path.join(data_dir_1, path)
        speaker_list = os.listdir(data_dir_2)
        parallel_set = {}

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

        # Print
        print(s+'_'+e)
        for utterance in parallel_set:
            print_list = list()
            for speaker in parallel_set[utterance]:
                # speaker: 'p235', speaker[1:]: '235'
                speaker_num = speaker[1:]
                sex = id_sex[speaker_num]

                print_content = speaker + '_' + sex
                print_list.append(print_content)

            print('%s: %s'%(utterance, print_content) )
