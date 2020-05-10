# %%
from performance_measure import msd_cal, mcd_cal, extract_ms
from tools.tools import load_pickle
from speech_tools import *
import librosa



# %%
mcd_list = list()
for file in sorted(os.listdir('processed_validation/inset_dev/p225/')):
    print(file)
    file_dir = os.path.join('processed_validation/inset_dev/p225/', file)
    coded_sp, ap, f0 = load_pickle(file_dir)
    decoded_sp = world_decode_mc(mc=coded_sp, fs= 16000)
    wav = world_speech_synthesis(f0=f0, decoded_sp=decoded_sp, ap=ap, fs=16000, frame_period=5.0 )
    soundfile.write('p225_001.wav', wav, 16000)

    wav, _ = librosa.load('p225_001.wav', sr=16000, mono=True)
    wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
    f0, timeaxis, sp, ap, mc = world_decompose(wav=wav, fs=16000, frame_period=5.0)

    mcd = mcd_cal(coded_sp, mc)
    mcd_list.append(mcd)
mcd_list
sum(mcd_list)/len(mcd_list)
# %%
os.listdir('trash')
wav, _ = librosa.load('trash/FM p225_008 p226_008.wav', sr=16000, mono=True)
wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
f0, timeaxis, sp, ap, mc1 = world_decompose(wav=wav, fs=16000, frame_period=5.0)

wav, _ = librosa.load('trash/p226_008.wav', sr=16000, mono=True)
wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
f0, timeaxis, sp, ap, mc2 = world_decompose(wav=wav, fs=16000, frame_period=5.0)

mcd = mcd_cal(mc1, mc2)
msd = msd_cal(mc1, mc2, 'vector')
print(mcd, msd)

wav, _ = librosa.load('trash/FF p228_006 p225_006.wav', sr=16000, mono=True)
wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
f0, timeaxis, sp, ap, mc1 = world_decompose(wav=wav, fs=16000, frame_period=5.0)

wav, _ = librosa.load('trash/p225_006.wav', sr=16000, mono=True)
wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
f0, timeaxis, sp, ap, mc2 = world_decompose(wav=wav, fs=16000, frame_period=5.0)

# mcd = mcd_cal(mc1, mc2)
msd = msd_cal(mc1, mc2, 'vector')
print(mcd, msd)

# %%
mcd = mcd_cal(coded_sp, coded_sp)
mcd = mcd_cal(mc, mc)

# %%
coded_sp, ap, f0 = load_pickle('processed_validation/inset_dev/p225/p225_001.p')
coded_sp.shape
decoded_sp = world_decode_mc(mc=coded_sp, fs= 16000)
wav = world_speech_synthesis(f0=f0, decoded_sp=decoded_sp, ap=ap, fs=16000, frame_period=5.0 )
soundfile.write('p225_001_selfconverted.wav', wav, 16000)

wav, _ = librosa.load('trash/p225_001_selfconverted.wav', sr=16000, mono=True)
wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
f0, timeaxis, sp, ap, mc = world_decompose(wav=wav, fs=16000, frame_period=5.0)
mcd = mcd_cal(coded_sp, mc)
msd = msd_cal(coded_sp, mc)
msd
mcd
from dtw import dtw
from performance_measure import dtw as dtw2
d1, p1 =dtw2(coded_sp[:,1:], mc[:,1:], dist = lambda x,y: 10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt(np.sum((x-y)**2, axis=1)))
p1[1][-1]
p1[0][-1]
d1/len(p1[0])
p1[1]
len(p1[0])
len(path[0])
coded_sp.shape
len(p1[0])
p1[0]
p1[1]
result[3][1]
result[3][0]
d, m, m, path = dtw(coded_sp[:,1:], mc[:,1:], dist = lambda x,y: 10.0 / np.log(10) * np.sqrt(2.0) * np.sqrt(np.sum((x-y)**2)))
d/len(path[0])
path[0].
p1
list([1])
d1 / len(path[0])
mcd
len(wav)

import matplotlib.pyplot as plt
fig, ax =plt.subplots(figsize=(10,10))
ax.matshow(mc, aspect='auto')
fig
# %%
fig1, ax1 =plt.subplots(figsize=(10,10))
ax1.matshow(mc, aspect='auto')
# fig1
fig2, ax2 =plt.subplots(figsize=(10,10))
ax2.matshow(coded_sp, aspect='auto')
# fig2
mc.shape


# %%


# self = Experiment(num_speakers=4)
#
# o = self.optimizer['VAE']
# o.param_groups[0]['lr']
# optimizer_new = optim.Adam(self.SC.parameters())
# # %%
# st= time.time()
# optimizer_new.state_dict()['param_groups'][0]['lr']
# et = time.time()
# print(et-st)
# st= time.time()
# optimizer_new.param_groups[0]['lr']
# et = time.time()
# print(et-st)
# coded_sp = np.ascontiguousarray(coded_sp)
# decoded_sp = world_decode_mc(mc = coded_sp, fs = 16000)
# wav = world_speech_synthesis(f0=f0, decoded_sp=decoded_sp, ap=ap, fs=16000, frame_period=5.0)
# coded_sp.shape
# f0.shape
# ap.shape
# plt.plot(wav)
# wav1 = np.nan_to_num(wav)
# librosa.output.write_wav(os.path.join(test_converted_dir, 'test.wav'), wav, sr=16000)
# import soundfile
# soundfile.write(os.path.join(test_converted_dir, 'test.wav'), data=wav, samplerate=16000)
# help(soundfile.write)
# help(librosa.output.write_wav)
# test_converted_dir
# loss_AC
# del loss_AC_A
# loss_ACforAC.backward
#
#         loss_AC_A = self.loss_ACforAC(x_batch_A, c_A)
#         loss_AC_B = self.loss_ACforAC(x_batch_B, c_B)
# loss_AC_A.backward()
# loss_AC_B.backward()
# t = loss_AC_A + loss_AC_B
# loss_ACforAC.backward()
# t.backward()
# testgrad=self.optimizer_AC.param_groups[0]['params'][0].grad
# self.optimizer_AC.param_groups[0]['params'][0].grad
# testgrad


# self
# self1 = self
# self1

# n
# help(np.random.seed)
# np.random.randint(10, size= 10)
# np.random.seed(0)
# torch.random.seed()
# help(torch.random.seed)
# torch.random.seed(0)
#
# self.Encoder = cc(Encoder(label_num = self.num_speakers))
# self.Encoder.conv1.weight
#
# help(np.random.randint)
#

# self.lr_scheduler_VAE.last_epoch
# self.lr_scheduler_VAE.state_dict()
# self.lr_scheduler_VAE.get_last_lr()
# self.lr_scheduler_VAE.get_lr()
#
# 0.00001==1e-5
# help(self.lr_scheduler_VAE.step)
# for i in range(1, 300+1):
#     print('epoch: %s, lr1: %s, lr2: %s, lr3: %s'%(i,self.optimizer_VAE.param_groups[0]['lr'], self.lr_scheduler_VAE.get_last_lr(), self.lr_scheduler_VAE.get_lr()))
#     self.optimizer_VAE.step()
#     self.lr_scheduler_VAE.step()

# w = torch.tensor([1,2,3,4,5], dtype = float, requires_grad = True)
# a = torch.tensor([2,2,2,2,2], dtype = float)
#
# h1=a*w
# h1
# h1_ = h1.sum()
# h1_.backward()
# h2=h1*w
# h2
# h2_=h2.sum()
# h2_
# h2_.backward()
# h3 = h2*w
# h3_ = h3.sum()
# h3_.backward()
# w.grad
# w.grad.data.zero_()
#
# w
# w_[0] = 7
# z = torch.as_tensor(w_)
# z
# y =a*w
# y_ = y.sum()
# y_.backward()
# a.grad
# b = a * 3
# c = a * 5
# c
# b_ = b.sum()
# b_
# b_.backward()
# c_ = c.sum()
# c_.backward()
# a.grad
# a.grad
# d = b+c
# d_ = d.sum()
# a.grad.data.zero_()
# a.grad
# d_.backward()
# e = a * 10
# e_ = e.sum()
# e_.backward()
#
# =0
# b = a+1
#

#
# from scipy.spatial.distance import euclidean
# def norm(p):
#     return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)
# x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
# y = np.array([[2,2], [3,3], [4,4]])
# x=coded_sps_norm_A[0].T
# y=coded_sps_norm_B[0].T
# x.shape
# y.shape
# x.shape
# y.shape
# distance1, path1 = fastdtw.fastdtw(x.T, y.T, dist=2)
# # %% time
# s1 = time.time()
# distance2, path2 = fastdtw.dtw(x,y, dist=1)
# e1 = time.time()
# s2 = time.time()
# distance4, c, ac, path4 = dtw.dtw(x,y,dist = norm(1))
# e2 = time.time()
#
# for p1, p2 in zip(path2, path4[1]):
#     print(p1[1], p2)
# len(path2)
# path4
#
# #  %%
# distance3, path3 = fastdtw.fasatdtw
# print(distance2, e1-s1)
# print(distance4, e2-s2)
#
# mcd1 = 10/np.log(10)*np.sqrt(2)*distance1
#
# print(mcd1)
# help(dtw.dtw)
# print(distance1)
# print(path1)
# print(distance2)
# print(path2)
#
# np.linalg.norm([1,2],2)
# help(np.atleast_1d)

# x_src = x_batch_A
# x_trg = x_batch_B
# si_src = si_batch_A
# si_trg = si_batch_B
# label_src = i
# label_trg = j

# si_A = load_pickle('processed_sis_train/p226/sis_train.p')
# for si in si_A:
#     print(si.shape)
# dataset_A.shape
# dataset_B.shape
# siset_A.shape
# for si in siset_A:
#     print(si.shape)
# len(coded_sps_norm_A)
# len(coded_sps_norm_B)
# len(si_A)
# len(si_B)
# trg_speaker
# train_data_B_dir
# coded_sps_norm_A[0].shape
# for c, d in zip(coded_sps_norm_A, si_A):
#     print(c.shape, d.shape)
# for c, d in zip(coded_sps_norm_B, si_B):
#     print(c.shape, d.shape)
#
# si_A
# len(si_B)
# si_A[0].shape
# si_A[1].shape
# file_dir = 'G:\\vctk\\inset_train\\p288'
# file_list = sorted(os.listdir(file_dir))
# file_list
# si_B = list()
# for file in file_list:
#     file_dir_1 = os.path.join(file_dir, file)
#     si = load_pickle(file_dir_1)
#     si_B.append(si)
#
# for si in si_B:
#     print(si.shape)
# a
# p=si_A[0]
# p.shape
# for p, c in zip(si_A, coded_sps_norm_A):
#     print(p.shape, c.shape)

    # def decode_step(self, z, index):
    #     mu, logvar = self.Decoder[index](z)
    #     return mu, logvar
    #
    # def clf_asr_step(self, x_src,x_trg,label_src,label_trg):
    #     '''Get Speaker Classifier loss & Automatic Speech Recognizer loss
    #     x_src - Encoder - Speaker Classifier - loss
    #     x_src - Encoder - Automatic Speech Recognizer - loss
    #     '''
    #     # Encoder
    #     mu_en,lv_en = self.encode_step(x_src,c_src)
    #     z = self.reparameterize(mu_en,lv_en)
    #     # KLD = self.KLD_loss(mu_en,lv_en)
    #
    #     # Speaker Classifier
    #     clf_logits = self.SC(z)
    #     clf_loss = self.clf_CrossEnt_loss(clf_logits,c)
    #
    #     # Automatic Speech Recognizer
    #     asr_logits = self.ASR(z)
    #     asr_loss = self.entropy_loss(asr_logits)
    #     # loss = self.CrossEnt_loss(logits, si_src)
    #
    #     # same_xt_mu,same_xt_lv = self.decode_step(z, c_src,label_src)
    #     # same_x_tilde = self.reparameterize(same_xt_mu,same_xt_lv)
    #     return clf_loss, asr_loss
    #
    # def vae_step(self, x_src,x_trg,label_src,label_trg):
    #     '''Get Variational AutoEncoder loss
    #     x_src - Encoder - loss
    #     x_src - Encoder - Decoder - loss
    #     '''
    #     c_src = self.generate_label(label_src, self.train_p['batch_size'])
    #     c_trg = self.generate_label(label_trg, self.train_p['batch_size'])
    #     c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
    #     c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
    #
    #     # Encoder
    #     mu_en,lv_en = self.encode_step(x_src,c_src)
    #     z = self.reparameterize(mu_en,lv_en)
    #
    #     # Decoder
    #     mu_xt,xt_lv = self.decode_step(z, c_trg, label_trg)
    #     x_tilde = self.reparameterize(mu_xt,xt_lv)
    #
    #     ###loss
    #     KLD = self.KLD_loss(mu_en,lv_en)
    #     loss_rec = -self.GaussianLogDensity(x_src,mu_xt,xt_lv) # Maximize probability
    #     return KLD,loss_rec,x_tilde
    #
    # def cycle_step(self, x_src, x_trg, si_src, si_target, label_src, label_trg):
    #     '''Get Cycle loss
    #     x_src - Encoder - Decoder - x_converted - Encoder - Decoder - x_reconstructed - loss
    #     x_src - Encoder - Decoder - x_converted - Encoder - loss
    #     '''
    #     c_src = self.generate_label(label_src,self.train_p['batch_size'])
    #     c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
    #     c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
    #     c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
    #
    #     # Encoder
    #     mu_en, lv_en = self.encode_step(x_src, c_src)
    #     z = self.reparameterize(mu_en, lv_en)
    #     # Decoder
    #     convert_xt_mu,convert_xt_lv = self.decode_step(z, c_trg, label_trg)
    #     convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
    #
    #     # Cycle
    #     # Encoder
    #     mu_en_cyc,lv_en_cyc = self.encode_step_cyc(convert_x_tilde, c_trg)
    #     z_cyc = self.reparameterize(mu_en_cyc,lv_en_cyc)
    #     # Decoder
    #     cyc_xt_mu,cyc_xt_lv = self.decode_step(z_cyc, c_src,label_src)
    #     cyc_x_tilde = self.reparameterize(cyc_xt_mu,cyc_xt_lv)
    #
    #     # Loss
    #     cyc_loss_rec = -self.GaussianLogDensity(x_src,cyc_xt_mu,cyc_xt_lv)
    #     cyc_KLD = self.KLD_loss(mu_en_cyc,lv_en_cyc)
    #
    #     return cyc_KLD, cyc_loss_rec
    #
    # def sem_step(self, x_src,x_trg,si_src,si_target,label_src,label_trg):
    #     c_src = self.generate_label(label_src,self.train_p['batch_size'])
    #     c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
    #     c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
    #     c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
    #
    #     # Encoder
    #     mu_en,lv_en = self.encode_step(x_src,c_src)
    #     z = self.reparameterize(mu_en,lv_en)
    #
    #     # Decoder
    #     convert_xt_mu,convert_xt_lv = self.decode_step(z, c_trg,label_trg)
    #     convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
    #
    #     # Cycle
    #     # Encoder
    #     mu_en_cyc,lv_en_cyc = self.encode_step_cyc(convert_x_tilde,c_trg)
    #     z_cyc = self.reparameterize(mu_en_cyc,lv_en_cyc)
    #
    #     # Mu? Z?
    #     # KLD_same_check = torch.mean(torch.abs(mu_en - mu_en_cyc))
    #     KLD_same_check = torch.mean((z - z_cyc)**2)
    #     return KLD_same_check
    #
    # def AC_step(self, x_src, x_trg, label_src,label_trg):
    #     '''Get Auxiliary Classifier loss
    #     x_src - Auxiliary Classifier - loss
    #     x_trg - Auxiliary Classifier - loss
    #     '''
    #     c_src = self.generate_label(label_src,self.train_p['batch_size'])
    #     c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
    #     c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
    #     c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
    #
    #     acc_s,src_t_label = self.AC(x_src)
    #     acc_t,trg_t_label = self.AC(x_trg)
    #
    #     AC_source =  self.CrossEnt_loss(src_t_label, c_src)
    #     AC_target =  self.CrossEnt_loss(trg_t_label, c_trg)
    #     return AC_source,AC_target
    #
    # def AC_F_step(self, x_src, x_trg, label_src, label_trg):
    #     '''Get Full Auxiliary Classifier loss
    #     x_src - Auxiliary Classifier - loss
    #     x_trg - Auxiliary Classifier - loss
    #     '''
    #     c_src = self.generate_label(label_src,self.train_p['batch_size'])
    #     c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
    #
    #     # Encoder
    #     mu_en,lv_en = self.encode_step(x_src,c_src)
    #     z = self.reparameterize(mu_en,lv_en)
    #
    #     # AC layer
    #     acc_s,t_label = self.AC(x_src)
    #     AC_real =  self.CrossEnt_loss(t_label, c_src)
    #
    #     # Decoder step - Full
    #     AC_cross_list = list()
    #     for i in range(self.num_speakers):
    #         c_trg = self.generate_label(i,self.train_p['batch_size'])
    #         c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
    #         # Decoder
    #         convert_xt_mu, convert_xt_lv = self.decode_step(z, c_trg,i)
    #         convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
    #         acc_conv_t, c_label = self.AC(convert_x_tilde)
    #         AC_cross = self.CrossEnt_loss(c_label, c_trg)
    #         AC_cross_list.append(AC_cross)
    #     AC_cross = sum(AC_cross_list) / self.num_speakers # Mean loss
    #
    #     return AC_real,AC_cross

    # if ((iteration+1) % self.train_p['iter_per_ep'])!=0 :
    #     # Update Speaker Clssifier (SC) module
    #     self.grad_reset()
    #     loss_SI_A = self.loss_SI(x_batch_A, c_onehot_A, c_A)
    #     loss_SI_B = self.loss_SI(x_batch_B, c_onehot_B, c_B)
    #     loss_SC = loss_SI_A + loss_SI_B
    #     loss_SC.backward()
    #     self.optimizer_SC.step()
    #
    # if ((iteration+1) % self.train_p['iter_per_ep'])==0 :
    #     # Update Automatic Speach Recognizer (ASR) module
    #     self.grad_reset()
    #     loss_LI_A = self.loss_LI(x_batch_A, c_onehot_A, si_batch_A)
    #     loss_LI_B = self.loss_LI(x_batch_B, c_onehot_B, si_batch_B)
    #     loss_ASR = loss_LI_A + loss_LI_B
    #     loss_ASR.backward()
    #     self.optimizer_ASR.step()
    #
    #     # Update Auxiliary Classifier (AC) module
    #     self.grad_reset()
    #     AC_source, AC_target = self.AC_step(x_batch_A, x_batch_B, i, j, self.train_p['batch_size'])
    #     AC_t_loss = AC_source + AC_target
    #     AC_t_loss.backward()
    #     self.optimizer_AC.step()
    #     self.grad_reset()
    #
    #     # Updat Variational AutoEncoder (VAE) module
    #     ###VAE step
    #     src_KLD, src_same_loss_rec, _= self.vae_step(x_batch_A, x_batch_B, i, i, self.train_p['batch_size'])
    #     trg_KLD, trg_same_loss_rec, _= self.vae_step(x_batch_B, x_batch_A, j, j, self.train_p['batch_size'])
    #     ###AC F step
    #     AC_real_src, AC_cross_src = self.AC_F_step(x_batch_A,x_batch_B,i,j,self.train_p['batch_size'])
    #     AC_real_trg, AC_cross_trg = self.AC_F_step(x_batch_B,x_batch_A,j,i,self.train_p['batch_size'])
    #     ###clf asr step
    #     clf_loss_A, asr_loss_A = self.clf_asr_step(x_batch_A,x_batch_B,i,j,self.train_p['batch_size'])
    #     clf_loss_B, asr_loss_B = self.clf_asr_step(x_batch_B,x_batch_A,j,i,self.train_p['batch_size'])
    #     CLF_loss = (clf_loss_A + clf_loss_B) / 2.0
    #     ASR_loss = (asr_loss_A + asr_loss_B) / 2.0
    #     ###Cycle step
    #     src_cyc_KLD, src_cyc_loss_rec = self.cycle_step(x_batch_A,x_batch_B,si_batch_A,si_batch_B,i,j,self.train_p['batch_size'])
    #     trg_cyc_KLD, trg_cyc_loss_rec = self.cycle_step(x_batch_B,x_batch_A,si_batch_B,si_batch_A,j,i,self.train_p['batch_size'])
    #     ###Semantic step
    #     src_semloss = self.sem_step(x_batch_A,x_batch_B,si_batch_A,si_batch_B,i,j,self.train_p['batch_size'])
    #     trg_semloss = self.sem_step(x_batch_B,x_batch_A,si_batch_B,si_batch_A,j,i,self.train_p['batch_size'])
    #     AC_f_loss = (AC_real_src + AC_real_trg + AC_cross_src + AC_cross_trg) / 4.0
    #     Sem_loss = (src_semloss + trg_semloss) / 2.0
    #     Cycle_KLD_loss = (src_cyc_KLD + trg_cyc_KLD) / 2.0
    #     Cycle_rec_loss = (src_cyc_loss_rec + trg_cyc_loss_rec) / 2.0
    #     KLD_loss = (src_KLD + trg_KLD)
    #     Rec_loss = (src_same_loss_rec + trg_same_loss_rec)
    #     loss = Rec_loss + KLD_loss + Cycle_KLD_loss + Cycle_rec_loss + AC_f_loss + Sem_loss - CLF_loss#+ASR_loss
    #     loss.backward()
    #     self.optimizer_VAE.step()
    #
    #     # Monitor Loss
    #     loss_result.loc[loss_result_index] = loss_SI, loss_LI, loss_AC, loss_Recon, loss_KLD, loss_Sem, loss_Cycle
# from utils import Hps
# from utils import Logger
# from utils import DataLoader
# from utils import to_var
# from utils import reset_grad
# from utils import multiply_grad
# from utils import grad_clip
# from utils import cal_acc
# from utils import calculate_gradients_penalty
# from utils import gen_noise
