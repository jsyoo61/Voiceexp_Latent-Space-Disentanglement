import os
import time

from speech_tools import *

data_dir_1 = os.path.join('../../../corpus')
data_dir_2 = 'inset/inset_dev'
data_dir_3 = os.path.join(data_dir_1, data_dir_2)
store_dir =  os.path.join('processed_eval')

sr = 16000
num_mcep = 36
frame_period = 5.0

speaker_list = sorted(os.listdir(data_dir_3))
start_time = time.time()

for speaker in speaker_list:
    data_dir = os.path.join(data_dir_3, speaker, '*.wav')
    flist = sorted(glob.glob(data_dir))

    exp_store_dir =  os.path.join(store_dir, speaker)
    os.makedirs(exp_store_dir, exist_ok=True)

    for file in flist:
        print(file)

        wav, _ = librosa.load(file, sr=sr, mono=True)
        wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
        f0, timeaxis, sp, ap, mc = world_decompose(wav=wav, fs=sr, frame_period=frame_period)
        mc = mc.T

        save_pickle(os.path.join(exp_store_dir, '{}.p'.format(file.split('/')[-1].split('.')[0])), (mc, ap, f0))

end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')
print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
