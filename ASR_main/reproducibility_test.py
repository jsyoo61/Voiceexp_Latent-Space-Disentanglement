import os
import numpy as np
import torch
from experiment import Experiment
from tools.tools import load_pickle, append
from tools.data import sort_load
from speech_tools import sample_train_data

self = Experiment(num_speakers=4)

# i=0
# j=1
# src_speaker = self.speaker_list[i]
# trg_speaker = self.speaker_list[j]
# train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
# train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
# si_A_dir = os.path.join(self.dirs['si'], src_speaker)
# si_B_dir = os.path.join(self.dirs['si'], trg_speaker)
#
# _, coded_sps_norm_A, _, _, _, _ = load_pickle(train_data_A_dir) #filelist_A, coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A
# _, coded_sps_norm_B, _, _, _, _ = load_pickle(train_data_B_dir)
# si_A = sort_load(si_A_dir) # [si, si, ...], si.shape == (n,)
# si_B = sort_load(si_B_dir)
#
# dataset_A, dataset_B, siset_A, siset_B, shuffled_id_A, shuffled_id_B = sample_train_data(dataset_A=coded_sps_norm_A, dataset_B=coded_sps_norm_B, ppgset_A=si_A, ppgset_B=si_B, n_frames=self.train_p['n_train_frames'])
# num_data = dataset_A.shape[0]
# append(self.dirs['train_id_log'], str(shuffled_id_A))
#
# dataset_A = np.expand_dims(dataset_A, axis=1)
# dataset_B = np.expand_dims(dataset_B, axis=1)
# c_A = self.generate_label(i, self.train_p['mini_batch_size']).to(self.device, dtype = torch.long)
# c_B = self.generate_label(j, self.train_p['mini_batch_size']).to(self.device, dtype = torch.long)
# c_onehot_A = self.label2onehot(c_A).to(self.device, dtype = torch.float)
# c_onehot_B = self.label2onehot(c_B).to(self.device, dtype = torch.float)
#
# start=0
# end=8
# x_batch_A = torch.as_tensor(dataset_A[start:end], device = self.device, dtype=torch.float)
# x_batch_B = torch.as_tensor(dataset_B[start:end], device = self.device, dtype=torch.float)
# si_batch_A = torch.as_tensor(siset_A[start:end], device = self.device, dtype=torch.long)
# si_batch_B = torch.as_tensor(siset_B[start:end], device = self.device, dtype=torch.long)
#
# loss_KLD_A, loss_rec_A = self.loss_MDVAE(x_batch_A, c_onehot_A)
# loss_KLD_B, loss_rec_B = self.loss_MDVAE(x_batch_B, c_onehot_B)
# loss_MDVAE = loss_KLD_A + loss_rec_A + loss_KLD_B + loss_rec_B

append(os.path.join(self.dirs['exp'], 'Encoder_'+str(0)+'.txt'), str(next(self.Encoder.parameters())) )
# self.optimizer['VAE'].zero_grad()
# loss_MDVAE.backward()
# self.optimizer['VAE'].step()
self.step()
append(os.path.join(self.dirs['exp'], 'Encoder_'+str(1)+'.txt'), str(next(self.Encoder.parameters())) )
