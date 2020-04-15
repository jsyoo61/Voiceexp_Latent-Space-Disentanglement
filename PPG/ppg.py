import os, sys
import net
import torch
import pickle as pk
import numpy as np
from tools import kaldi_manager as km
from tools import data_manager as dm

model_path="model/timit_sp_ppg_mono"
model = net.BiGRU_HMM(feat_dim=36, hidden_dim=256, num_layers=5, dropout=0.2, output_dim=144)
model.load_state_dict(torch.load(model_path+"/final.pt"))
model.cuda()
model.eval()

with open(model_path+"/prior.pk", 'rb') as f:
    prior = pk.load(f)

for stype in ["inset", "outset"]:
    for dtype in ["train", "dev", "test"]:
        data_id = stype+"_"+dtype
        print('Processing data_id:%s'%(data_id))

        ########################################################
        feat_path = "data/vctk/"+data_id
        with open(feat_path+"/sp.pk", 'rb') as f:
            feat_dict = pk.load(f)

        ########################################################

        for utt_id, feat_mat in feat_dict.items():
            print(utt_id)
            spk_id = utt_id.split("_")[0]
            x = torch.Tensor(feat_mat).float().cuda()
            with torch.no_grad():
                result = model([x])
            result = result.detach().cpu().numpy()
            ppg = result - prior

            result_path="result/vctk/"+data_id+"/"+spk_id
            os.makedirs(result_path, exist_ok=True)
            with open(result_path+"/"+utt_id+".pk", 'wb') as f:
                pk.dump(ppg, f)
print('Process Complete.')
