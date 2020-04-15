import os, sys
import pickle as pk
import numpy as np
from tools.tools import load_pickle, save_pickle

for stype in ["inset", "outset"]:
    for dtype in ["train", "dev", "test"]:
        data_id = stype+"_"+dtype
        print('Processing data_id:%s'%(data_id))

        ########################################################
        ppg_path_0 = os.path.join('result/vctk/', data_id)
        save_path_0 = os.path.join('result/state_index', data_id)
        spk_ids = os.listdir(ppg_path_0)
        ########################################################
        for spk_id in spk_ids:
            ppg_path_1 = os.path.join(ppg_path_0, spk_id)
            save_path_1 = os.path.join(save_path_0, spk_id)
            os.makedirs(save_path_1, exist_ok=True)
            ppg_filelist = os.listdir(ppg_path_1)

            for ppg_file in ppg_filelist:
                ppg_path = os.path.join(ppg_path_1, ppg_file)
                save_path = os.path.join(save_path_1, ppg_file)

                ppg = load_pickle(ppg_path)
                state_index = ppg.argmax(axis = 1)
                save_pickle(state_index, save_path)

print('Process Complete.')
