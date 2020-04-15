# %% Initialize
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
from performance_measure import extract_ms, mcd_cal, msd_cal, gv_cal
from loss import *
from utils import cc
from speech_tools import sample_train_data, transpose_in_list, world_decompose
from tools.tools import load_pickle, save_pickle, readlines, read, append, Printer
from tools.data import sort_load

# # Dummy class for debugging
# class Dummy():
#     """ Dummy class for debugging """
#     def __init__(self):
#         pass
# self = Dummy()
# self.dirs = dict()
# exp_dir = 'exp/'
# model_dir = 'model/'
# log_dir = 'log.txt'
# train_data_dir = 'processed/'
# si_dir = 'processed_stateindex/'
# test_data_dir = 'processed_validation/inset_dev/'
# test_pathlist_dir = 'filelist/in_dev.lst'
# self.dirs['exp'] = exp_dir
# model_dir = os.path.join(exp_dir, model_dir)
# self.dirs['model'] = model_dir
# log_dir = os.path.join(self.dirs['exp'], log_dir)
# self.dirs['log'] = log_dir
# self.dirs['train_data'] = train_data_dir
# self.dirs['si'] = si_dir
# self.dirs['test_data'] = test_data_dir
# self.dirs['test_pathlist'] = test_pathlist_dir
# self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
# self.num_speakers = len(self.speaker_list)
# self.train_p = dict()
# self.train_p['batch_size'] = 8
# self.train_p['n_epoch'] = 100
# self.train_p['iter_per_ep'] = 4
# test_pathlist = read(test_pathlist_dir).splitlines()
# test_path = test_pathlist[0]
# i=0
# j=2
# src_speaker = self.speaker_list[i]
# trg_speaker = self.speaker_list[j]
#
# torch.manual_seed(0)
# self
# self1 = self
# self1

# %%
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

class Experiment(object):
    '''
    Data descriptors:
    dirs
        Hold all directory information to be used in the experiment
        in dict() form.
        To see all directories that are related, refer to:
        list(self.dirs.items())
    model_p
        Hold all parameters related to NN model learning in dict() form
    train_p
        Hold all parameters related to training in dict() form
    '''
    def __init__(self, num_speakers = 100, exp_name = None, model_p = None, new = True):
        # 1] Initialize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(0)
        torch.manual_seed(0)
        self.create_env(exp_name, new)
        # self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
        self.speaker_list = ['p225','p226','p227','p228']
        self.num_speakers = len(self.speaker_list)
        assert self.num_speakers == num_speakers, 'Specified "num_speakers" and "num_speakers in train data" does not match'
        self.build_model(params = model_p)
        self.model_p = model_p
        self.p = Printer(filewrite_dir = self.dirs['log'])

        # 2] Hyperparameters for Training
        self.train_p = dict()
        self.train_p['n_train_frames'] = 128
        self.train_p['iter_per_ep'] = 4
        self.train_p['start_epoch'] = 1
        self.train_p['n_epoch'] = 100
        self.train_p['batch_size'] = 8

        self.preprocess_p = dict(
        sr = 16000,
        frame_period = 5.0,
        num_mcep = 36,
        )

        # 3] Hyperparameters for saving model
        self.model_kept = []
        self.max_keep=100

        # If the experiment is not new, Load most recent model
        if new == False:
            model_list = os.listdir(self.dirs['model'])
            most_trained_model = max(model_list)
            epoch_trained = int(most_trained_model.split('-')[-1])
            self.train_p['start_epoch'] += epoch_trained
            print('Loading model from %s'%most_trained_model)
            self.load_model(os.path.join(self.dirs['model'], most_trained_model))

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
        test_pathlist_dir = 'filelist/inset_dev.lst'
        validation_dir = 'validation/'
        validation_result_dir = 'validation_result/'
        validation_log_dir = 'validation_log.txt'

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
        model_dir = os.path.join(self.dirs['exp'], model_dir)
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

        # 6] Validation
        validation_dir = os.path.join(self.dirs['exp'], validation_dir)
        os.makedirs(validation_dir, exist_ok=True)
        self.dirs['validation'] = validation_dir
        validation_result_dir = os.path.join(self.dirs['validation'], validation_result_dir)
        os.makedirs(validation_result_dir, exist_ok=True)
        self.dirs['validation_result'] = validation_result_dir
        validation_log_dir = os.path.join(self.dirs['validation'], validation_log_dir)
        self.dirs['validation_log'] = validation_log_dir

    def set_train_param(self, train_param):
        for param, value in train_param.items():
            self.train_p[param] = value

    def build_model(self, params = None):
        '''
        Initialize NN model & optimizers
        '''
        self.Encoder = cc(Encoder(label_num = self.num_speakers))
        self.Decoder = [cc(Decoder()) for i in range(self.num_speakers)]
        self.SC = cc(SpeakerClassifier(label_num = self.num_speakers))
        self.ASR = cc(AutomaticSpeechRecognizer())
        self.AC = cc(AuxiliaryClassifier(label_num = self.num_speakers))

        # if params == None:
        #     params = dict(
        #     vae_lr = 0.001,
        #     vae_betas = (0.9,0.999),
        #     sc_lr = 0.0002,
        #     sc_betas = (0.5,0.999),
        #     asr_lr = 0.00001,
        #     asr_betas = (0.5,0.999),
        #     ac_lr = 0.00005,
        #     ac_betas = (0.5,0.999),
        #     )

        decoder_parameter_list = list()
        for decoder in self.Decoder:
            decoder_parameter_list += list(decoder.parameters())
        vae_params = list(self.Encoder.parameters()) + decoder_parameter_list

        self.optimizer_VAE = optim.Adam(vae_params, lr=params['vae_lr'], betas=params['vae_betas'])
        self.optimizer_SC = optim.Adam(self.SC.parameters(), lr=params['sc_lr'], betas=params['sc_betas'])
        self.optimizer_ASR = optim.Adam(self.ASR.parameters(), lr=params['asr_lr'], betas=params['asr_betas'])
        self.optimizer_AC = optim.Adam(self.AC.parameters(), lr=params['ac_lr'], betas=params['ac_betas'])

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

        new_model_path = os.path.join(model_path,'epoch_{}.pt'.format(epoch))
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

    def load_whole_model(self,model_path):
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
        self.optimizer_SC.zero_grad()
        self.optimizer_ASR.zero_grad()
        self.optimizer_AC.zero_grad()
        self.optimizer_VAE.zero_grad()

    def generate_label(self, index, batch_size):
        '''label == [index, index, ...] with len(label) == self.train_p['batch_size'] '''
        labels = [index] * int(batch_size)
        return torch.tensor(labels)

    def label2onehot(self, labels):
        # labels: [1, 1, 1, ...], with len(labels) == batch_size
        batch_size = len(labels)
        onehot = torch.zeros(batch_size, self.num_speakers)
        onehot[:, labels.long()] = 1 # labels itself are indices for the onehot vectors, ONLY WORKS IF "Labels" ARE ALL EQUAL!! (With differnt labels, that syntax does not work)
        return onehot

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def test_step(self, x, c_src, c_trg, trg_speaker_num):
        self.set_eval(trg_speaker_num)
        with torch.no_grad(): # Enabled for faster computation
            mu, logvar = self.Encoder(x,c_src)
            xt_mu,xt_logvar = self.Decoder[trg_speaker_num](mu)
        return xt_mu.detach().cpu().numpy()

    def loss_MDVAE(self, x, c_onehot):
        z_mu, z_logvar = self.Encoder(x, c_onehot)
        loss_KLD = KLD_loss(z_mu,z_logvar)
        z = self.reparameterize(z_mu, z_logvar)
        # Decoder - All
        loss_rec_list = list()
        for i in range(self.num_speakers):
            xt_mu, xt_logvar = self.Decoder[i](z)
            loss_rec = -GaussianLogDensity(x,xt_mu,xt_logvar) # Maximize probability
            loss_rec_list.append(loss_rec)
        loss_rec = sum(loss_rec_list) / self.num_speakers # Mean loss
        # loss_MDVAE = loss_KLD + loss_rec
        # return loss_MDVAE
        return loss_KLD, loss_rec

    def loss_SI(self, x, c_onehot, c):
        '''
        Get Speaker Classifier loss
        x_src - Encoder - Speaker Classifier - loss
        '''
        mu, logvar = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.SC(z)
        loss_SI = F.cross_entropy(logits,c)
        return loss_SI

    def loss_LI(self, x, c_onehot, si):
        '''
        Get Automatic Speech Recognizer loss
        x_src - Encoder - Automatic Speech Recognizer - loss
        '''
        mu, logvar = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.ASR(z)
        loss_LI = F.cross_entropy(logits, si)
        return loss_LI

    # def loss_AC(self, x, c_onehot_src, )

    def decode_step(self, z, index):
        mu, logvar = self.Decoder[index](z)
        return mu, logvar

    def clf_asr_step(self, x_src,x_trg,label_src,label_trg):
        '''Get Speaker Classifier loss & Automatic Speech Recognizer loss
        x_src - Encoder - Speaker Classifier - loss
        x_src - Encoder - Automatic Speech Recognizer - loss
        '''
        c = self.generate_label(label_src,self.train_p['batch_size']).to(self.device, dtype = torch.long)
        c_src = self.generate_label(label_src,self.train_p['batch_size'])
        # c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
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

    def vae_step(self, x_src,x_trg,label_src,label_trg):
        '''Get Variational AutoEncoder loss
        x_src - Encoder - loss
        x_src - Encoder - Decoder - loss
        '''
        c_src = self.generate_label(label_src, self.train_p['batch_size'])
        c_trg = self.generate_label(label_trg, self.train_p['batch_size'])
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

    def cycle_step(self, x_src, x_trg, si_src, si_target, label_src, label_trg):
        '''Get Cycle loss
        x_src - Encoder - Decoder - x_converted - Encoder - Decoder - x_reconstructed - loss
        x_src - Encoder - Decoder - x_converted - Encoder - loss
        '''
        c_src = self.generate_label(label_src,self.train_p['batch_size'])
        c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
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

    def sem_step(self, x_src,x_trg,si_src,si_target,label_src,label_trg):
        c_src = self.generate_label(label_src,self.train_p['batch_size'])
        c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
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

    def AC_step(self, x_src, x_trg, label_src,label_trg):
        '''Get Auxiliary Classifier loss
        x_src - Auxiliary Classifier - loss
        x_trg - Auxiliary Classifier - loss
        '''
        c_src = self.generate_label(label_src,self.train_p['batch_size'])
        c_trg = self.generate_label(label_trg,self.train_p['batch_size'])
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        acc_s,src_t_label = self.AC(x_src)
        acc_t,trg_t_label = self.AC(x_trg)

        AC_source =  self.CrossEnt_loss(src_t_label, c_src)
        AC_target =  self.CrossEnt_loss(trg_t_label, c_trg)
        return AC_source,AC_target

    def AC_F_step(self, x_src, x_trg, label_src, label_trg):
        '''Get Full Auxiliary Classifier loss
        x_src - Auxiliary Classifier - loss
        x_trg - Auxiliary Classifier - loss
        '''
        c_src = self.generate_label(label_src,self.train_p['batch_size'])
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
            c_trg = self.generate_label(i,self.train_p['batch_size'])
            c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
            # Decoder
            convert_xt_mu, convert_xt_lv = self.decode_step(z, c_trg,i)
            convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
            acc_conv_t, c_label = self.AC(convert_x_tilde)
            AC_cross = self.CrossEnt_loss(c_label, c_trg)
            AC_cross_list.append(AC_cross)
        AC_cross = sum(AC_cross_list) / self.num_speakers # Mean loss

        return AC_real,AC_cross

    def train(self, lambd, train_param = None, train_data_dir = None, model_dir = None):
        # 0] Manual Directory&Parmeter designation
        if train_data_dir is not None:
            self.dirs['train_data'] = train_data_dir
        if model_dir is not None:
            self.dirs['model'] = model_dir
        if train_param is not None:
            self.set_train_param(train_param)

        # 0] Hyperparameters - preprocess
        validation_summary = pd.DataFrame(columns = ['mcd', 'msd', 'gv'])
        self.dirs['validation_summary'] = os.path.join(self.dirs['validation'], 'validation_summary.p')

        # 1] Print Train Settings
        p = Printer(filewrite_dir = self.dirs['log']) # Printer prints out to stdout & log
        self.p.add('Start Training with the following setting:\n')
        self.p.add('torch version: %s\n'%torch.__version__)
        self.p.add('train_data_dir: %s\n'%self.dirs['train_data'])
        self.p.add('model_dir: %s\n'%self.dirs['model'])
        self.p.add('num_speakers: %s\n'%self.num_speakers)
        self.p.add('training parameters:\n%s\n'%self.train_p)
        self.p.add('lambd: %s\n'%lambd)
        self.p.add('model optimizer configurations:\n')
        self.p.add('VAE: %s\n'%self.optimizer_VAE.defaults)
        self.p.add('SC: %s\n'%self.optimizer_SC.defaults)
        self.p.add('ASR: %s\n'%self.optimizer_ASR.defaults)
        self.p.add('AC: %s\n'%self.optimizer_AC.defaults)
        self.p.print(end='')
        # self.p.add('n_epoch: %s\n'%self.train_p['n_epoch'])
        # self.p.add('batch_size:%s\n'%self.train_p['batch_size'])

        # 3] Start training
        for ep in range(self.train_p['start_epoch'], self.train_p['start_epoch'] + self.train_p['n_epoch']):
            self.p.print('Epoch:%s'%ep)
            np.random.seed(0)
            # Keep track of loss results for monitoring
            loss_result_list = list()
            # loss_result = pd.DataFrame(index = range(self.num_speakers * self.train_p['iter_per_ep']), columns = loss_result_list)
            time_start = time.time()
            for i, src_speaker in enumerate(self.speaker_list):
                for j, trg_speaker in enumerate(self.speaker_list):

                    # Load train data, and si
                    # Source: A, Target: B
                    train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
                    train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
                    si_A_dir = os.path.join(self.dirs['si'], src_speaker)
                    si_B_dir = os.path.join(self.dirs['si'], trg_speaker)

                    _, coded_sps_norm_A, _, _, _, _ = load_pickle(train_data_A_dir) #filelist_A, coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A
                    _, coded_sps_norm_B, _, _, _, _ = load_pickle(train_data_B_dir)
                    si_A = sort_load(si_A_dir) # [si, si, ...], si.shape == (n,)
                    si_B = sort_load(si_B_dir)

                    dataset_A, dataset_B, siset_A, siset_B = sample_train_data(dataset_A=coded_sps_norm_A, dataset_B=coded_sps_norm_B, ppgset_A=si_A, ppgset_B=si_B, n_frames=self.train_p['n_train_frames'])
                    num_data = dataset_A.shape[0]

                    dataset_A = np.expand_dims(dataset_A, axis=1)
                    dataset_B = np.expand_dims(dataset_B, axis=1)
                    siset_A = np.expand_dims(siset_A, axis=1)
                    siset_B = np.expand_dims(siset_B, axis=1)
                    # dataset_A = torch.as_tensor(dataset_A, device = self.device, dtype=torch.float)
                    # dataset_B = torch.as_tensor(dataset_B, device = self.device, dtype=torch.float)
                    # siset_A = torch.as_tensor(siset_A, device = self.device, dtype=torch.float)
                    # siset_B = torch.as_tensor(siset_B, device = self.device, dtype=torch.float)
                    c_A = self.generate_label(i, self.train_p['batch_size']).to(self.device, dtype = torch.long)
                    c_B = self.generate_label(j, self.train_p['batch_size']).to(self.device, dtype = torch.long)
                    c_onehot_A = self.label2onehot(c_A).to(self.device, dtype = torch.float)
                    c_onehot_B = self.label2onehot(c_B).to(self.device, dtype = torch.float)

                    # p.print('source: %s, target: %s, num_data: %s'%(src_speaker, trg_speaker, num_data))

                    for iteration in range(self.train_p['iter_per_ep']):
                        start = iteration * self.train_p['batch_size']
                        end = (iteration + 1) * self.train_p['batch_size']

                        x_batch_A = torch.as_tensor(dataset_A[start:end], device = self.device, dtype=torch.float)
                        x_batch_B = torch.as_tensor(dataset_B[start:end], device = self.device, dtype=torch.float)
                        si_batch_A = torch.as_tensor(siset_A[start:end], device = self.device, dtype=torch.float)
                        si_batch_B = torch.as_tensor(siset_B[start:end], device = self.device, dtype=torch.float)

                        self.grad_reset()
                        loss_KLD_A, loss_rec_A = self.loss_MDVAE(x_batch_A, c_onehot_A)
                        loss_KLD_B, loss_rec_B = self.loss_MDVAE(x_batch_B, c_onehot_B)
                        loss_MDVAE = loss_KLD_A + loss_rec_A + loss_KLD_B + loss_rec_B
                        loss_MDVAE.backward()
                        self.optimizer_VAE.step()

                        loss_result_list.append(loss_MDVAE)

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
            time_end = time.time()
            time_elapsed = time_end - time_start
            # loss_result_index = ['SI','LI','AC','Recon','KLD','Sem','Cycle']
            loss_result_index = ['MDVAE']
            loss_result = pd.DataFrame(loss_result_list, columns = loss_result_index)

            self.p.print('Time elapsed this epoch: {:0.1f}m {:0.5}s'.format( time_elapsed // 60, time_elapsed % 60 ))
            self.p.print('Mean\n'+ str(loss_result.mean().to_frame().T))
            self.p.print('Std\n' + str(loss_result.std().to_frame().T))

            # Save model
            if (ep) % 1 ==0:
                self.p.print("Saving Model, epoch: {}".format(ep))
                self.save_model(self.dirs['model'], ep)

            # Validation
            # if (ep) % 50 ==0:
            if (ep) % 1 ==0:
                # 1. Get performance measures
                validation_result_dir = os.path.join(self.dirs['validation_result'], 'validation_{}.p'.format(ep))
                validation_result = self.performance_measure()

                # 2. Save results
                save_pickle(validation_result, validation_result_dir)
                validation_result_mean = validation_result.mean()
                validation_summary.loc[ep] = validation_result_mean.values
                save_pickle(validation_summary, self.dirs['validation_summary'])

                # 3. print
                self.p.print("Validation Complete.")
                self.p.print('Mean\n' + str(validation_result.mean().to_frame().T))
                self.p.print('Std\n' + str(validation_result.std().to_frame().T))
                append(self.dirs['validation_log'], '{} {} {} {}\n'.format(ep, validation_result_mean['mcd'],validation_result_mean['msd'],validation_result_mean['gv']))
                for measure in validation_summary.columns:
                    fig_save_dir = os.path.join(self.dirs['validation'], measure+'.png')
                    axes = validation_summary.plot(y = measure, style='o-')
                    fig = axes.get_figure()
                    fig.savefig(fig_save_dir)
                self.set_train()

    def performance_measure(self, test_data_dir = None, test_pathlist_dir = None):
        if test_data_dir is not None:
            self.dirs['test_data'] = test_data_dir
        if test_pathlist_dir is not None:
            self.dirs['test_pathlist'] = test_pathlist_dir

        test_pathlist = read(self.dirs['test_pathlist']).splitlines()

        # 1] Convert all data
        print('Loading&Converting test data...')
        converted_mcep_list = list()
        target_mcep_list = list()
        tested_pathlist = list()
        start_time = time.time()
        np.random.seed(0)
        np.random.shuffle(test_pathlist)
        for speaker_A in self.speaker_list:
            for speaker_B in self.speaker_list:
                n_sample = 0
                if speaker_A == speaker_B:
                    continue
                for test_path in test_pathlist:
                    if n_sample >= 10:
                        break
                    # Specify conversion details
                    conversion_path_sex, filename_src, filename_trg = test_path.split()
                    src_speaker = filename_src.split('_')[0]
                    trg_speaker = filename_trg.split('_')[0]
                    ##### FOR TESTING, ERASE LATER
                    # if src_speaker not in self.speaker_list or trg_speaker not in self.speaker_list:
                    if src_speaker != speaker_A or trg_speaker != speaker_B:
                        continue
                    n_sample += 1

                    label_src = self.speaker_list.index(src_speaker)
                    label_trg = self.speaker_list.index(trg_speaker)

                    # Define datapath
                    data_A_dir = os.path.join(self.dirs['test_data'], src_speaker, '{}.p'.format(filename_src))
                    data_B_dir = os.path.join(self.dirs['test_data'], trg_speaker, '{}.p'.format(filename_trg))
                    train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
                    train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))

                    # Load data
                    _, coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, _, _ = load_pickle(train_data_A_dir)
                    _, coded_sps_norm_B, coded_sps_mean_B, coded_sps_std_B, _, _ = load_pickle(train_data_B_dir)
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
                        self.p.print('shape not coherent?? file:%s'%(filename_src))
                        self.p.print(test_path)
                        coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]

                    coded_sp_converted = coded_sp_converted_norm * coded_sps_std_B + coded_sps_mean_B
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                    converted_mcep_list.append(coded_sp_converted)
                    target_mcep_list.append(coded_sp_trg)
                    tested_pathlist.append(test_path)


        # 2] Calculate performance_measures (MCD, MSD, GV)
        print('Calculating MCD, MSD, GV')
        pool = Pool(30)
        converted_ms_list = pool.map(extract_ms, converted_mcep_list)
        target_ms_list = pool.map(extract_ms, target_mcep_list)
        pool.close()
        pool.join()

        pool = Pool(30)
        mcd_list = pool.starmap(mcd_cal, zip(converted_mcep_list, target_mcep_list))
        msd_list = pool.starmap(msd_cal, zip(converted_ms_list, target_ms_list))
        gv_list = pool.starmap(gv_cal, zip(converted_mcep_list))
        pool.close()
        pool.join()

        # 3] Save Results
        test_result = pd.DataFrame(index = tested_pathlist, columns = ['mcd', 'msd', 'gv'], dtype = float)
        print('Calculation complete.')
        for test_path, mcd, msd, gv in zip(tested_pathlist, mcd_list, msd_list, gv_list):
            test_result.loc[test_path] = mcd, msd, gv

        end_time = time.time()
        time_elapsed = end_time - start_time
        self.p.print('Time elapsed for testing: {:0.1f}m {:0.5}s'.format( time_elapsed // 60, time_elapsed % 60 ))
        return test_result

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
