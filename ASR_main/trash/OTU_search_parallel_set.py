import os
import numpy as np
import pickle

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

set_type = ['inset', 'outset']
eval_type = ['dev', 'test']
result = dict()
for s in set_type:
    result[s] = dict()
    for t in eval_type:
        result[s][t] = list()

data_dir = '../../corpus'
parallel_set = load_pickle('parallel_set.p')
for u in parallel_set:
    print('%s:%s'%(u, parallel_set[u]))
# Search each utterances
for utterance in parallel_set:

    print('utterance:%s'%(utterance))

    # All speakers that spoke this utterance
    speakers = parallel_set[utterance]

    # Find out where "speaker_utterance" exists
    for speaker in speakers:
        file = speaker + '_' + utterance + '.wav'
        found = False
        print('file:%s'%(file))
        for s in set_type:

            if found == True:
                break

            data_dir_1 = os.path.join(data_dir, s)

            for t in eval_type:

                if found == True:
                    break

                path = s + '_' + t
                data_dir_2 = os.path.join(data_dir_1, path)

                data_dir_3 = os.path.join(data_dir_2, speaker)

                if os.path.exists(data_dir_3):
                    file_list = os.listdir(data_dir_3)
                    # print(file_list)
                    if file in file_list:
                        result[s][t].append(file)
                        found = True
                else:
                    print('no directory: %s'%(data_dir_3))

        if found == False:
            print('utterance:%s, speaker:%s, file:%s, not found!!'%(utterance, speaker, file))


with open('parallel_set.txt','w') as f:
    for s in result:
        for t in result[s]:
            content = s + '_' + t
            content += str(result[s][t])
            content += '\n'
            f.write(content)

            print('set:%s, type:%s'%(s, t))
            for file in result[s][t]:
                print(file)
