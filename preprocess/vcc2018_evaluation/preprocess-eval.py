import os
import time

from speech_tools import *

data_dir_1 = os.path.join('../../../corpus')
store_dir_1 =  os.path.join('processed_eval')

sr = 16000
num_mcep = 36
frame_period = 5.0

start_time = time.time()
for stype in ["inset", "outset"]:
    for dtype in ["dev", "test"]:
        data_id = stype+"_"+dtype
        print('Processing data_id:%s'%(data_id))

        data_dir_2 = os.path.join(data_dir_1, stype, data_id)
        store_dir_2 = os.path.join(store_dir_1, data_id)
        speaker_list = sorted(os.listdir(data_dir_2))

        for speaker in speaker_list:
            print(speaker)
            data_dir = os.path.join(data_dir_2, speaker, '*.wav')
            flist = sorted(glob.glob(data_dir))

            exp_store_dir =  os.path.join(store_dir_2, speaker)
            os.makedirs(exp_store_dir, exist_ok=True)

            for file in flist:
                wav, _ = librosa.load(file, sr=sr, mono=True)
                wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
                f0, timeaxis, sp, ap, mc = world_decompose(wav=wav, fs=sr, frame_period=frame_period)

                save_pickle(os.path.join(exp_store_dir, '{}.p'.format(file.split('/')[-1].split('.')[0])), (mc, ap, f0))
end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')
print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed // 60), (time_elapsed % 60 // 1)))
