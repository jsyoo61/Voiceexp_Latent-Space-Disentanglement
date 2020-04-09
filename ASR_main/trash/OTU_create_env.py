import os
os.chdir('ASR_main/')
from tools import load_pickle, save_pickle

data_dir = 'processed'
def save():
    speaker_list = os.listdir(data_dir)
    save_pickle(speaker_list, 'trash/speaker_list.p')

def load():
    speaker_list = load_pickle('speaker_list.p')
    os.makedirs(data_dir, exist_ok=True)
    for speaker in speaker_list:
        data_dir_1 = os.path.join(data_dir, speaker)
        os.makedirs(data_dir_1, exist_ok=True)

if __name__ == '__main__':
    save()
