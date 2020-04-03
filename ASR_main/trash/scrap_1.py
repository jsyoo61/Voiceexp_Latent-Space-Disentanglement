import os
os.chdir('ASR_main/')
import numpy as np
import pickle
os.getcwd()
os.listdir()

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# %% search parallel set
set_type = ['inset', 'outset']
eval_type = ['dev', 'test']
result = dict()
for s in set:
    result[s] = dict()
    for t in type:
        result[s][t] = list

data_dir = '../../corpus'
parallel_set = load_pickle('parallel_set.p')
for u in parallel_set:
    print(parallel_set[u])
# Search each utterances
for utterance in parallel_set:

    # All speakers that spoke this utterance
    speakers = parallel_set[utterance]

    # Find out where "speaker_utterance" exists
    for speaker in speakers:
        file = speaker + '_' + utterance
        found = False
        for s in set:
            data_dir_1 = os.path.join(data_dir, s)
            for t in type:
                path = s + '_' + t
                data_dir_2 = os.path.join(data_dir_1, path)

                data_dir_3 = os.path.join(data_dir_2, speaker)

                file_list = os.listdir(data_dir_3)
                if file in file_list:
                    result[s][t].append(file)

        if found = False:
            print('utterance:%s, speaker:%s, not found!!'%(utterance, speaker))

with open('parallel_set.txt','w') as f:
    for s in result:
        for t in result[s]:
            content = s + '_' + t
            content += result[s][t]
            f.write(content)


# %% make parallel_set
with open('inset_outset.txt', 'r') as f:
    inset_outset = f.readlines()

inset_outset
inset_outset_all = list()
for line in inset_outset:
    inset_outset_all.extend(line.split())

parallel_set = dict()
for utterance in inset_outset_all:
    utterance_split = utterance.split('_')

    utterance_num = utterance_split[-1]
    speaker = utterance_split[0]

    if utterance_num not in parallel_set:
        parallel_set[utterance_num] = list()
        parallel_set[utterance_num].append(speaker)
    else:
        parallel_set[utterance_num].append(speaker)

parallel_set
save_pickle('parallel_set.p', parallel_set)


len(text)
pair =[]
num_pair = []
# %% aa
for t in text:
    # print(t.split())
    pair.append(t.split())
t1 = t.split()
t1[0].split('_')[0][1:]
len(pair)
for p in pair:
    num_pair.append(len(p))
num_pair
pair
parallel_set = {}
for p in pair:
    utternace_num = p[0].split('_')[-1]
    parallel_set[utternace_num] = list()
    for file in p:
        speaker = file.split('_')[0]
        parallel_set[utternace_num].append(speaker)

num_pair
save_pickle('parallel_set.p', parallel_set)
ptest = load_pickle('parallel_set.p')
ptest

ppgs_test = load_pickle('processed_ppgs_train/p225/ppgs_train.p')
ppgs_test[0].shape
ppgs_test1 = load_pickle('p225_002.pk')
ppgs_test1.shape

# %% aa
with open('speaker-info.txt', 'r') as f:
    txt2 = f.readlines()
txt2
len(txt2)
id = []
sex = []
id_sex = {}
for l in txt2:
    data = l.split()
    data_id = data[0]
    data_sex = data[2]
    id.append(data_id)
    sex.append(data_sex)
    id_sex[data_id] = data_sex
id
sex
id_sex

save_pickle('id_sex.p', id_sex)
