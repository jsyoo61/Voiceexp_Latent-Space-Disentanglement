import os
import librosa
import kaldi_io

import numpy as np
import pickle as pk
from speech_tools import world_decompose

for stype in ["inset", "outset"]:
    for dtype in ["train", "dev", "test"]:
        data_id = stype+"_"+dtype
        print("Processing",data_id,"...")
        feat_dict = dict()

        wav_dir=os.path.join("wav/vctk/", stype, data_id)
        for root, _, files in os.walk(wav_dir):
            print(root)
            for wav_name in files:
                spk_id = wav_name.split("_")[0]
                utt_id = wav_name.split(".")[0]
                wav_path = os.path.join(root, wav_name)
                wav, _ = librosa.load(wav_path, sr=16000, mono=True)
                wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
                _, _, _, _, feat = world_decompose(wav=wav, fs=16000, frame_period=5.0, num_mcep=36)

                feat_dict[utt_id] = feat

        data_dir="data/vctk/"+data_id
        os.makedirs(data_dir, exist_ok=True)
        # data_path = os.path.join(data_dir, data_id)
        with open(data_dir+"/sp.pk", 'wb') as f:
            pk.dump(feat_dict, f)
print('Process Complete.')
