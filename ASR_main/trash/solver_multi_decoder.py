import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import pandas as pd

from multiprocessing import Pool
from model import Encoder, Decoder, AuxiliaryClassifier, AutomaticSpeechRecognizer, SpeakerClassifier
from performance_measure import extract_ms, mcd_cal, msd_cal
from loss import *
from utils import cc
from speech_tools import sample_train_data, transpose_in_list, world_decompose
from tools.tools import load_pickle, readlines, read, append, Printer, sort_load

from preprocess_MCD import *
from utils import Hps
from utils import Logger
from utils import DataLoader
from utils import to_var
from utils import reset_grad
from utils import multiply_grad
from utils import grad_clip
from utils import cal_acc
from utils import calculate_gradients_penalty
from utils import gen_noise

class Solver(object):
    '''

    Data descriptors:
    dirs
        Hold all directory information to be used in the experiment
        in dict() form.
        To see all directories that are related, refer to:
        list(self.dirs.items())
    '''
    def __init__(self, num_speakers = 100, exp_name = None, new = True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_env(exp_name, new)
        self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
        self.num_speakers = len(self.speaker_list)
        self.build_model()

        assert self.num_speakers == num_speakers, 'Specified "num_speakers" and "num_speakers in train data" does not match'

        self.model_kept = []
        self.max_keep=100

        # Hyperparameters
        self.n_training_frames = 128


    def create_env(self, exp_name = None, new = True):
        '''
        Create experiment environment
        Store all "static directories" required for experiment in "self.dirs"(dict)

        Store every experiment result in: exp/exp_name/ == exp_dir
        including log, model, test(validation) etc
        '''
        # 0] exp_dir == master directory
        self.dirs = dict()
        exp_dir = 'exp/'
        model_dir = 'model/'
        log_dir = 'log.txt'
        train_data_dir = 'processed/'
        si_dir = 'processed_stateindex/'
        test_data_dir = 'processed_validation/inset_dev/'
        test_pathlist_dir = 'filelist/in_dev.lst'

        # 1] Set up Experiment directory
        if exp_name == None:
            exp_name = time.strftime('%m%d_%H%M%S')
        exp_dir = os.path.join(exp_dir, exp_name)
        if new == True:
            assert not os.path.isdir(exp_dir), 'New experiment, but exp_dir with same name exists'
            os.makedirs(exp_dir)
        else:
            assert os.path.isdir(exp_dir), 'Existing experiment, but exp_dir doesn\'t exist'
        self.dirs['exp'] = exp_dir

        # 2] Model parameter directory
        model_dir = os.path.join(exp_dir, model_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.dirs['model'] = model_dir

        # 3] Log settings
        log_dir = os.path.join(self.dirs['exp'], log_dir)
        self.dirs['log'] = log_dir

        # 4] Train data
        self.dirs['train_data'] = train_data_dir
        self.dirs['si'] = si_dir

        # 5] Test (Including Validation)
        self.dirs['test_data'] = test_data_dir
        self.dirs['test_pathlist'] = test_pathlist_dir

    def build_model(self):
        '''
        Initialize NN model & optimizers
        '''
        self.Encoder = cc(Encoder(label_num = self.num_speakers))
        self.Decoder = [cc(Decoder(label_num = self.num_speakers)) for i in range(self.num_speakers)]
        self.AC = cc(AuxiliaryClassifier(label_num = self.num_speakers))
        self.ASR = cc(AutomaticSpeechRecognizer())
        self.SC = cc(SpeakerClassifier(label_num = self.num_speakers))
        ac_betas = (0.5,0.999)
        vae_betas = (0.9,0.999)
        ac_lr = 0.00005
        vae_lr = 0.001
        clf_betas = (0.5,0.999)
        asr_betas = (0.5,0.999)
        clf_lr = 0.0002
        asr_lr = 0.00001

        decoder_parameter_list = list()
        for decoder in self.Decoder:
            decoder_parameter_list += list(decoder.parameters())
        vae_params = list(self.Encoder.parameters()) + decoder_parameter_list

        self.ac_optimizer = optim.Adam(self.AC.parameters(), lr=ac_lr, betas=ac_betas)
        self.vae_optimizer = optim.Adam(vae_params, lr=vae_lr, betas=vae_betas)
        self.asr_optimizer = optim.Adam(self.ASR.parameters(), lr=asr_lr, betas=asr_betas)
        self.clf_optimizer = optim.Adam(self.SC.parameters(), lr=clf_lr, betas=clf_betas)

    def save_model(self, model_path, epoch,enc_only=True):
        all_model=dict()
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
            }
        else:
            all_model['encoder'] = self.Encoder.state_dict()

            for i, decoder in enumerate(self.Decoder):
                module_name = 'decoder_' + str(i)
                all_model[module_name] = decoder.state_dict()

            all_model['AC'] = self.AC.state_dict()

        new_model_path = os.path.join(model_path,'{}-{}'.format(model_path, epoch))
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path, speaker_num, enc_only=True):
        speaker_num = int(speaker_num)
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            decoder_name = 'decoder_'+str(speaker_num)
            self.Decoder[speaker_num].load_state_dict(all_model[decoder_name])

    def load_whole_model(self,model_path,enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])

            self.Decoder.load_state_dict(all_model['decoder'])

    def set_train(self):
        self.Encoder.train()
        for decoder in self.Decoder:
            decoder.train()

    def set_eval(self,trg_speaker_num):
        self.Encoder.eval()
        self.Decoder[trg_speaker_num].eval()

    def grad_reset(self):
        self.ac_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        self.asr_optimizer.zero_grad()
        self.clf_optimizer.zero_grad()

    def generate_label(self, index, batch_size):
        '''label == [index, index, ...] with len(label) == batch_size '''
        labels = [index] * int(batch_size)
        return torch.tensor(labels)

    def label2onehot(self, labels):
        # labels: [1, 1, 1, ...], with len(labels) == batch_size
        batch_size = len(labels)
        # labels = torch.tensor(labels)
        onehot = torch.zeros(batch_size, self.num_speakers)
        onehot[:, labels.long()] = 1 # labels itself are indices for the onehot vectors
        return onehot

    def test_step(self, x, c_src,c_trg,trg_speaker_num, gen=False):
        self.set_eval(trg_speaker_num)
        # Encoder
        mu_en,lv_en = self.Encoder(x,c_src)
        z = self.reparameterize(mu_en,lv_en)
        # Decoder
        xt_mu,xt_lv = self.Decoder[trg_speaker_num](mu_en, c_trg)
        # x_tilde = self.reparameterize(xt_mu,xt_lv)
        return xt_mu.detach().cpu().numpy()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_SI(self, x, c, c_onehot):
        '''
        Get Speaker Classifier loss
        x_src - Encoder - Speaker Classifier - loss
        '''
        mu, logvar = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.SC(z)
        loss_SI = F.cross_entropy(logits,c)
        return loss_SI

    def encode_step(self, x,c):
        mu, logvar = self.Encoder(x,c)
        return mu, logvar

    def encode_step_cyc(self, x,c):
        mu, logvar  = self.Encoder(x,c)
        return mu, logvar

    def decode_step(self, z, c, index):
        mu, logvar = self.Decoder[index](z, c)
        return mu, logvar

    def clf_step(self, x_src, label, batch_size):
        '''Get Speaker Classifier loss
        x_src - Encoder - Speaker Classifier - loss
        '''
        # Generate label
        c = self.generate_label(label,batch_size).to(device = self.device, dtype = torch.long) # [label, label, ...]
        c_src = self.generate_label(label,batch_size)
        c_src = self.label2onehot(c_src).to(device = self.device, dtype=torch.float) # label -> onehot vector

        # Encoder
        mu_en, lv_en = self.encode_step(x_src, c_src)
        z = self.reparameterize(mu_en, lv_en)

        # Speaker Classifier
        logits = self.SC(z)

        loss = self.clf_CrossEnt_loss(logits,c)
        return loss

    def asr_step(self, x_src, si_src, label, batch_size):
    # def asr_step(self, x_src, label, batch_size):
        '''Get Automatic Speech Recognizer loss
        x_src - Encoder - Automatic Speech Recognizer - loss
        '''
        # Generate label
        c_src = self.generate_label(label, batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)

        # Encoder
        mu_en, lv_en = self.encode_step(x_src, c_src)
        z = self.reparameterize(mu_en, lv_en)

        # Automatic Speech Recognizer
        logits = self.ASR(z)

        loss = self.entropy_loss(logits)
        # loss = self.CrossEnt_loss(logits, si_src)
        return loss

    def clf_asr_step(self, x_src,x_trg,label_src,label_trg,batch_size):
        '''Get Speaker Classifier loss & Automatic Speech Recognizer loss
        x_src - Encoder - Speaker Classifier - loss
        x_src - Encoder - Automatic Speech Recognizer - loss
        '''
        c = self.generate_label(label_src,batch_size).to(self.device, dtype = torch.long)
        c_src = self.generate_label(label_src,batch_size)
        # c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        # c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)
        # KLD = self.KLD_loss(mu_en,lv_en)

        # Speaker Classifier
        clf_logits = self.SC(z)
        clf_loss = self.clf_CrossEnt_loss(clf_logits,c)

        # Automatic Speech Recognizer
        asr_logits = self.ASR(z)
        asr_loss = self.entropy_loss(asr_logits)
        # loss = self.CrossEnt_loss(logits, si_src)

        # same_xt_mu,same_xt_lv = self.decode_step(z, c_src,label_src)
        # same_x_tilde = self.reparameterize(same_xt_mu,same_xt_lv)
        return clf_loss, asr_loss

    def vae_step(self, x_src,x_trg,label_src,label_trg,batch_size):
        '''Get Variational AutoEncoder loss
        x_src - Encoder - loss
        x_src - Encoder - Decoder - loss
        '''
        c_src = self.generate_label(label_src, batch_size)
        c_trg = self.generate_label(label_trg, batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)

        # Decoder
        xt_mu,xt_lv = self.decode_step(z, c_trg, label_trg)
        x_tilde = self.reparameterize(xt_mu,xt_lv)

        ###loss
        KLD = self.KLD_loss(mu_en,lv_en)
        loss_rec = -self.GaussianLogDensity(x_src,xt_mu,xt_lv) # Maximize probability
        return KLD,loss_rec,x_tilde

    def cycle_step(self, x_src, x_trg, si_src, si_target, label_src, label_trg, batch_size):
        '''Get Cycle loss
        x_src - Encoder - Decoder - x_converted - Encoder - Decoder - x_reconstructed - loss
        x_src - Encoder - Decoder - x_converted - Encoder - loss
        '''
        c_src = self.generate_label(label_src,batch_size)
        c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en, lv_en = self.encode_step(x_src, c_src)
        z = self.reparameterize(mu_en, lv_en)
        # Decoder
        convert_xt_mu,convert_xt_lv = self.decode_step(z, c_trg, label_trg)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)

        # Cycle
        # Encoder
        mu_en_cyc,lv_en_cyc = self.encode_step_cyc(convert_x_tilde, c_trg)
        z_cyc = self.reparameterize(mu_en_cyc,lv_en_cyc)
        # Decoder
        cyc_xt_mu,cyc_xt_lv = self.decode_step(z_cyc, c_src,label_src)
        cyc_x_tilde = self.reparameterize(cyc_xt_mu,cyc_xt_lv)

        # Loss
        cyc_loss_rec = -self.GaussianLogDensity(x_src,cyc_xt_mu,cyc_xt_lv)
        cyc_KLD = self.KLD_loss(mu_en_cyc,lv_en_cyc)

        return cyc_KLD, cyc_loss_rec

    def sem_step(self, x_src,x_trg,si_src,si_target,label_src,label_trg,batch_size):
        c_src = self.generate_label(label_src,batch_size)
        c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)

        # Decoder
        convert_xt_mu,convert_xt_lv = self.decode_step(z, c_trg,label_trg)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)

        # Cycle
        # Encoder
        mu_en_cyc,lv_en_cyc = self.encode_step_cyc(convert_x_tilde,c_trg)
        z_cyc = self.reparameterize(mu_en_cyc,lv_en_cyc)

        # Mu? Z?
        # KLD_same_check = torch.mean(torch.abs(mu_en - mu_en_cyc))
        KLD_same_check = torch.mean((z - z_cyc)**2)
        return KLD_same_check

    def AC_step(self, x_src, x_trg, label_src,label_trg,batch_size):
        '''Get Auxiliary Classifier loss
        x_src - Auxiliary Classifier - loss
        x_trg - Auxiliary Classifier - loss
        '''
        c_src = self.generate_label(label_src,batch_size)
        c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        acc_s,src_t_label = self.AC(x_src)
        acc_t,trg_t_label = self.AC(x_trg)

        AC_source =  self.CrossEnt_loss(src_t_label, c_src)
        AC_target =  self.CrossEnt_loss(trg_t_label, c_trg)
        return AC_source,AC_target

    def AC_F_step(self, x_src, x_trg, label_src, label_trg, batch_size):
        '''Get Full Auxiliary Classifier loss
        x_src - Auxiliary Classifier - loss
        x_trg - Auxiliary Classifier - loss
        '''
        c_src = self.generate_label(label_src,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)

        # AC layer
        acc_s,t_label = self.AC(x_src)
        AC_real =  self.CrossEnt_loss(t_label, c_src)

        # Decoder step - Full
        AC_cross_list = list()
        for i in range(self.num_speakers):
            c_trg = self.generate_label(i,batch_size)
            c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
            # Decoder
            convert_xt_mu, convert_xt_lv = self.decode_step(z, c_trg,i)
            convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
            acc_conv_t, c_label = self.AC(convert_x_tilde)
            AC_cross = self.CrossEnt_loss(c_label, c_trg)
            AC_cross_list.append(AC_cross)
        AC_cross = sum(AC_cross_list) / self.num_speakers # Mean loss

        return AC_real,AC_cross

    def train(self, batch_size, lambd, n_epoch = None, train_data_dir = None, model_dir = None):
        # 0] Manual directory designation
        if train_data_dir is not None:
            self.dirs['train_data'] = train_data_dir
        if model_dir is not None:
            self.dirs['model'] = model_dir

# num_speakers = 100
# batch_size = 8
# train_data_dir = 'processed'
#
# self = Solver()
# self.num_speakers
        # 0] Hyperparameters - preprocess
        sr = 16000
        frame_period = 10.0
        num_mcep = 36

        # 1] Experiment settings
        if n_epoch is None:
            n_epoch = 100

        # 2] Log settings
        p = Printer(filewrite_dir = self.dirs['log']) # Printer prints out to stdout & log

# i=0
# j=2
# src_speaker = self.speaker_list[i]
# trg_speaker = self.speaker_list[j]
# src_speaker
# trg_speaker

        # 1. Start training
        if mode == 'ASR_TIMIT':
            for ep in range(1, n_epoch + 1):
                p.print('epoch:%s'%epoch)
                time_start = time.time()
                np.random.seed()
                for i, src_speaker in enumerate(self.speaker_list):
                    for j, trg_speaker in enumerate(self.speaker_list):

                        # Load train data, and si
                        # Source: A, Target: B
                        train_data_A_dir = os.path.join(train_data_dir, src_speaker, 'cache{}.p'.format(num_mcep))
                        train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(num_mcep))
                        si_A_dir = os.path.join(si_dir, src_speaker)
                        si_B_dir = os.path.join(si_dir, trg_speaker)

                        _, coded_sps_norm_A, _, _, _, _ = load_pickle(train_data_A_dir) #filelist_A, coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A
                        _, coded_sps_norm_B, _, _, _, _ = load_pickle(train_data_B_dir)
                        si_A = sort_load(si_A_dir) # [si, si, ...], si.shape == (n,)
                        si_B = sort_load(si_B_dir)
# p=si_A[0]
# p.shape
# for p, c in zip(si_A, coded_sps_norm_A):
#     print(p.shape, c.shape)

                        dataset_A, dataset_B, siset_A, siset_B = sample_train_data(dataset_A=coded_sps_norm_A, dataset_B=coded_sps_norm_B, siset_A=si_A, siset_B=si_B, n_frames=self.n_training_frames)
                        num_data = dataset_A.shape[0]

                        dataset_A = np.expand_dims(dataset_A, axis=1)
                        dataset_A = torch.as_tensor(dataset_A, device = self.device, dtype=torch.float)
                        dataset_B = np.expand_dims(dataset_B, axis=1)
                        dataset_B = torch.as_tensor(dataset_B, device = self.device, dtype=torch.float)
                        siset_A = np.expand_dims(siset_A, axis=1)
                        siset_A = torch.as_tensor(siset_A, device = self.device, dtype=torch.float)
                        siset_B = np.expand_dims(siset_B, axis=1)
                        siset_B = torch.as_tensor(siset_B, device = self.device, dtype=torch.float)
                        c_A = self.generate_label(i, batch_size).to(self.device, dtype = torch.long)
                        c_B = self.generate_label(j, batch_size).to(self.device, dtype = torch.long)
                        c_onehot_A = self.label2onehot(c_A).to(self.device, dtype = torch.long)
                        c_onehot_B = self.label2onehot(c_B).to(self.device, dtype = torch.long)

                        p.print('source: %s, target: %s, num_data: %s'%(src_speaker, trg_speaker, num_data))
                        for iteration in range(4):
                            start = iteration * batch_size
                            end = (iteration + 1) * batch_size

                            x_batch_A = dataset_A[start:end]
                            x_batch_B = dataset_B[start:end]
                            si_batch_A = siset_A[start:end]
                            si_batch_B = siset_B[start:end]

                            if ((iteration+1) % 4)!=0 :
                                # Update Speaker Clssifier (CLF) module
                                self.grad_reset()
                                # Generate label
                                c = self.generate_label(label,batch_size).to(device = self.device, dtype = torch.long) # [label, label, ...]
                                c_src = self.label2onehot(c_src).to(device = self.device, dtype=torch.float) # label -> onehot vector

                                # Encoder
                                mu_en, lv_en = self.encode_step(x_src, c_src)
                                z = self.reparameterize(mu_en, lv_en)

                                # Speaker Classifier
                                logits = self.SC(z)

                                loss = self.clf_CrossEnt_loss(logits,c)


                                clf_loss_A = self.clf_step(x_batch_A, i, batch_size)
                                clf_loss_B = self.clf_step(x_batch_B, j, batch_size)
                                CLF_loss = clf_loss_A + clf_loss_B
                                loss = CLF_loss
                                loss.backward()
                                self.clf_optimizer.step()

                            if ((iteration+1) % 4)==0 :
                                # Update Automatic Speach Recognizer (ASR) module
                                self.grad_reset()
                                asr_loss_A = self.asr_step(x_batch_A, i, batch_size)
                                asr_loss_B = self.asr_step(x_batch_B, j, batch_size)
                                asr_loss = asr_loss_A + asr_loss_B
                                loss = asr_loss
                                loss.backward()
                                self.asr_optimizer.step()

                                # Update Auxiliary Classifier (AC) module
                                self.grad_reset()
                                AC_source, AC_target = self.AC_step(x_batch_A, x_batch_B, i, j, batch_size)
                                AC_t_loss = AC_source + AC_target
                                AC_t_loss.backward()
                                self.ac_optimizer.step()
                                self.grad_reset()

                                ###VAE step
                                src_KLD, src_same_loss_rec, _= self.vae_step(x_batch_A, x_batch_B, i, i, batch_size)
                                trg_KLD, trg_same_loss_rec, _= self.vae_step(x_batch_B, x_batch_A, j, j, batch_size)

                                ###AC F step
                                AC_real_src, AC_cross_src = self.AC_F_step(x_batch_A,x_batch_B,i,j,batch_size)
                                AC_real_trg, AC_cross_trg = self.AC_F_step(x_batch_B,x_batch_A,j,i,batch_size)

                                ###clf asr step
                                clf_loss_A, asr_loss_A = self.clf_asr_step(x_batch_A,x_batch_B,i,j,batch_size)
                                clf_loss_B, asr_loss_B = self.clf_asr_step(x_batch_B,x_batch_A,j,i,batch_size)
                                CLF_loss = (clf_loss_A + clf_loss_B) / 2.0
                                ASR_loss = (asr_loss_A + asr_loss_B) / 2.0

                                ###Cycle step
                                src_cyc_KLD, src_cyc_loss_rec = self.cycle_step(x_batch_A,x_batch_B,si_batch_A,si_batch_B,i,j,batch_size)
                                trg_cyc_KLD, trg_cyc_loss_rec = self.cycle_step(x_batch_B,x_batch_A,si_batch_B,si_batch_A,j,i,batch_size)

                                ###Semantic step
                                src_semloss = self.sem_step(x_batch_A,x_batch_B,si_batch_A,si_batch_B,i,j,batch_size)
                                trg_semloss = self.sem_step(x_batch_B,x_batch_A,si_batch_B,si_batch_A,j,i,batch_size)

                                AC_f_loss = (AC_real_src + AC_real_trg + AC_cross_src + AC_cross_trg) / 4.0
                                Sem_loss = (src_semloss + trg_semloss) / 2.0
                                Cycle_KLD_loss = (src_cyc_KLD + trg_cyc_KLD) / 2.0
                                Cycle_rec_loss = (src_cyc_loss_rec + trg_cyc_loss_rec) / 2.0
                                KLD_loss = (src_KLD + trg_KLD)
                                Rec_loss = (src_same_loss_rec + trg_same_loss_rec)
                                loss = Rec_loss + KLD_loss + Cycle_KLD_loss + Cycle_rec_loss + AC_f_loss + Sem_loss - CLF_loss#+ASR_loss
                                loss.backward()
                                self.vae_optimizer.step()
                time_end = time.time()
                time_elapsed = time_end - time_start
                p.print('Time elapsed this epoch: %.1sm:%.5ss'%( time_elapsed // 60, time_elapsed % 60 ))
                p.print("Epoch : {}, Recon : {:.3f}, KLD : {:.3f}, AC t Loss : {:.3f}, AC f Loss : {:.3f}, Sem Loss : {:.3f}, Clf : {:.3f}, Asr Loss : {:.3f}"\
                    .format(ep,Rec_loss,KLD_loss,AC_t_loss,AC_cross_trg,Sem_loss,CLF_loss,ASR_loss))

                # Save model
                if (ep) % 50 ==0:
                    p.print("Model Save Epoch {}".format(ep))
                    self.save_model(self.dirs['model_dir'], ep)

                # Validation
                if (ep) % 50 ==0:
                    validation_result_dir = os.path.join(self.dirs['exp_dir'], 'validation_{}.p'.format(ep))
                    validation_result = self.performance_measure(output_suffix = ep)
                    save_pickle(validation_result, validation_result_dir)
                    p.print("Epoch {} : Validation Process Complete.".format(ep))
                    self.set_train()

    def performance_measure(self, test_data_dir = None, test_pathlist_dir = None, output_suffix = None):
        # mfcc_target, mfcc_hat, sr = 16000, frame_period = 10.0):
        if test_data_dir is not None:
            self.dirs['test_data'] = test_data_dir
        if test_pathlist_dir is not None:
            self.dirs['test_pathlist'] = test_pathlist_dir
        if output_suffix is not None:
            test_log_name = "test_{}.txt".format(output_suffix)
        else:
            test_log_name = "test.txt"

        test_log_dir = os.path.join(self.dirs['exp_dir'], test_log_name)
        p = Printer(filewrite_dir = test_log_dir)

        test_pathlist = read(self.dirs['test_pathlist']).splitlines()
        test_result = pd.DataFrame(index = test_pathlist, columns = ['mcd', 'msd', 'gv'], dtype = float)

# test_path = test_pathlist[0]
        # 1] Convert all data
        p.print('Loading&Converting test data...')
        converted_mcep_list = list()
        target_mcep_list = list()
        for test_path in test_pathlist:

            # Specify conversion details
            conversion_path_sex, filename_src, filename_trg = test_path.split()
            src_speaker = filename_src.split('_')[0]
            trg_speaker = filename_trg.split('_')[0]
            label_src = self.speaker_list.index(src_speaker)
            label_trg = self.speaker_list.index(trg_speaker)

            # Define datapath
            data_A_dir = os.path.join(self.dirs['test_data'], src_speaker, '{}.p'.format(filename_src))
            data_B_dir = os.path.join(self.dirs['test_data'], trc_speaker, '{}.p'.format(filename_trg))
            train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(num_mcep))
            train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(num_mcep))

            # Load data
            coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, _, _ = load_pickle(os.path.join(train_data_A_dir, 'cache{}.p'.format(num_mcep)))
            coded_sps_norm_B, coded_sps_mean_B, coded_sps_std_B, _, _ = load_pickle(os.path.join(train_data_B_dir, 'cache{}.p'.format(num_mcep)))
            coded_sp, _, f0 = load_pickle(data_A_dir) # coded_sp.shape: (T, 36)
            coded_sp_trg, _, _ = load_pickle(data_B_dir)

            # Prepare input
            coded_sp = coded_sp.T
            coded_sp_norm = (coded_sp - coded_sps_mean_A) / coded_sps_std_A
            coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
            coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
            coded_sp_norm = torch.from_numpy(coded_sp_norm).to(self.device, dtype=torch.float)

            c_src = self.generate_label(label_src,1)
            c_trg = self.generate_label(label_trg,1)
            c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
            c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

            # Convert
            coded_sp_converted_norm = self.test_step(coded_sp_norm, c_src, c_trg, label_trg)

            # Post-process output
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)

            # Additional conversions
            # f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A, mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
            if coded_sp_converted_norm.shape[1] > len(f0):
                p.print('shape not coherent?? file:%s'%(filename_src))
                p.print(test_path)
                coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]

            coded_sp_converted = coded_sp_converted_norm * coded_sps_std_B + coded_sps_mean_B
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

            converted_mcep_list.append(coded_sp_converted)
            target_mcep_list.append(coded_sp_trg)

        # 2] Calculate performance_measures (MCD, MSD, GV)
        p.print('Calculating MCD, MSD, GV')
        pool = Pool(35)
        converted_ms_list = pool.map(extract_ms, converted_mcep_list)
        target_ms_list = pool.map(extract_ms, target_mcep_list)
        pool.close()
        pool.join()

        pool = Pool(35)
        mcd_list = pool.starmap(mcd_cal, zip(converted_mcep_list, target_mcep_list))
        msd_list = pool.starmap(msd_cal, zip(converted_ms_list, target_ms_list))
        pool.close()
        pool.join()

        # 3] Save Results
        p.print('Calculation complete.')
        for test_path, mcd, msd, gv in zip(test_pathlist, mcd_list, msd_list, gv_list):
            test_result.loc[test_path] = mcd, msd, gv
            p.print(test_path + str(mcd) + ' ' + str(msd) + ' ' + str(gv) + '\n')

        return test_result

w = torch.tensor([1,2,3,4,5], dtype = float, requires_grad = True)
a = torch.tensor([2,2,2,2,2], dtype = float)

h1=a*w
h1
h1_ = h1.sum()
h1_.backward()
h2=h1*w
h2
h2_=h2.sum()
h2_
h2_.backward()
h3 = h2*w
h3_ = h3.sum()
h3_.backward()
w.grad
w.grad.data.zero_()

w
w_[0] = 7
z = torch.as_tensor(w_)
z
y =a*w
y_ = y.sum()
y_.backward()
a.grad
b = a * 3
c = a * 5
c
b_ = b.sum()
b_
b_.backward()
c_ = c.sum()
c_.backward()
a.grad
a.grad
d = b+c
d_ = d.sum()
a.grad.data.zero_()
a.grad
d_.backward()
e = a * 10
e_ = e.sum()
e_.backward()

=0
b = a+1


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
