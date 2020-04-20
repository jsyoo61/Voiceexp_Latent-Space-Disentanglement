# %% Initialize
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import platform

from multiprocessing import Pool
from model import Encoder, Decoder, AuxiliaryClassifier, AutomaticSpeechRecognizer, SpeakerClassifier
from performance_measure import extract_ms, mcd_cal, msd_cal, gv_cal
from loss import *
from utils import cc
from speech_tools import sample_train_data, transpose_in_list, world_decompose, world_decode_mc, pitch_conversion, world_speech_synthesis
from tools.tools import load_pickle, save_pickle, readlines, read, append, Printer, stars
from tools.data import sort_load

# %%
class Experiment(object):
    '''
    ------------------
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
    speaker_list
        Hold all name of speakers
    '''
    def __init__(self, num_speakers = 100, exp_name = None, new = True, model_p = None, train_p = None):
        np.random.seed(0)
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 0] Default parameters
        if model_p == None:
            model_p = dict(
            vae_lr = 0.0001,
            vae_betas = (0.9,0.999),
            sc_lr = 0.0002,
            sc_betas = (0.5,0.999),
            asr_lr = 0.00001,
            asr_betas = (0.5,0.999),
            ac_lr = 0.00005,
            ac_betas = (0.5,0.999),
            )
        self.model_p = model_p

        # 1] Initialize
        self.create_env(exp_name, new)
        # self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
        self.speaker_list = ['p225','p226','p227','p228']
        self.num_speakers = len(self.speaker_list)
        assert self.num_speakers == num_speakers, 'Specified "num_speakers" and "num_speakers in train data" does not match'
        self.build_model(params = model_p)
        self.p = Printer(filewrite_dir = self.dirs['log'])

        # 2] Hyperparameters for Training
        self.train_p = dict()
        self.train_p['n_train_frames'] = 128
        self.train_p['batch_size'] = 32
        self.train_p['mini_batch_size'] = 8
        self.train_p['iter_per_ep'] = self.train_p['batch_size'] // self.train_p['mini_batch_size']
        assert self.train_p['iter_per_ep'] * self.train_p['mini_batch_size'] == self.train_p['batch_size'], 'Specified batch_size "%s" cannot be divided by mini_batch_size "%s"'%(self.train_p['batch_size'], self.train_p['mini_batch_size'])
        self.train_p['start_epoch'] = 1
        self.train_p['n_epoch'] = 200
        self.train_p['model_save_epoch'] = 2
        self.train_p['validation_epoch'] = 2
        self.train_p['sample_per_path'] = 10
        self.lambd = dict(
        SI = 1,
        LI = 1,
        AC = 1,
        SC = 1,
        C = 1,
        )
        self.preprocess_p = dict(
        sr = 16000,
        frame_period = 5.0,
        num_mcep = 36,
        )
        self.p_measure_list = ['mcd', 'msd_all', 'msd_vector', 'gv']

        # 3] Hyperparameters for saving model
        self.model_kept = []
        self.max_keep=100

        # 4] If the experiment is not new, Load most recent model
        if new == False:
            self.model_kept= sorted(os.listdir(self.dirs['model']))
            most_trained_model = max(model_list)
            epoch_trained = int(most_trained_model.split('-')[-1])
            self.train_p['start_epoch'] += epoch_trained
            # Update lr_scheduler
            print('Loading model from %s'%most_trained_model)
            self.load_model_all(os.path.join(self.dirs['model'], most_trained_model))

    def create_env(self, exp_name = None, new = True):
        '''Create experiment environment
        Store all "static directories" required for experiment in "self.dirs"(dict)

        Store every experiment result in: exp/exp_name/ == exp_dir
        including log, model, test(validation) etc
        '''
        # 0] exp_dir == master directory
        self.dirs = dict()
        exp_dir = 'exp/'
        model_dir = 'model/'
        log_dir = 'log.txt'
        # Train
        train_data_dir = 'processed/'
        si_dir = 'processed_stateindex/'
        loss_log_dir = 'loss_log.txt'
        # Validation
        validation_data_dir = 'processed_validation/inset_dev/'
        validation_pathlist_dir = 'filelist/inset_dev.lst'
        validation_dir = 'validation/'
        validation_result_dir = 'validation_result/'
        validation_converted_dir = 'validation_converted/'
        validation_log_dir = 'validation_log.txt'
        # Test
        test_dir = 'test/'
        test_result_dir = 'test_result/'
        test_converted_dir = 'converted/'
        test_data_dir = 'processed_validation/inset_test/'
        test_pathlist_dir = 'filelist/inset_test.lst'

        # 1] Set up Experiment directory
        if exp_name == None:
            exp_name = time.strftime('%m%d_%H%M%S')
        self.dirs['exp'] = os.path.join(exp_dir, exp_name)
        if new == True:
            assert not os.path.isdir(self.dirs['exp']), 'New experiment, but exp_dir with same name exists'
            os.makedirs(self.dirs['exp'])
        else:
            assert os.path.isdir(self.dirs['exp']), 'Existing experiment, but exp_dir doesn\'t exist'

        # 2] Model parameter directory
        self.dirs['model'] = os.path.join(self.dirs['exp'], model_dir)
        os.makedirs(self.dirs['model'], exist_ok=True)

        # 3] Log settings
        self.dirs['log'] = os.path.join(self.dirs['exp'], log_dir)

        # 4] Train
        self.dirs['train_data'] = train_data_dir
        self.dirs['si'] = si_dir
        self.dirs['loss_log'] = os.path.join(self.dirs['exp'], loss_log_dir)

        # 6] Validation
        self.dirs['validation_data'] = validation_data_dir
        self.dirs['validation_pathlist'] = validation_pathlist_dir
        self.dirs['validation'] = os.path.join(self.dirs['exp'], validation_dir)
        self.dirs['validation_result'] = os.path.join(self.dirs['validation'], validation_result_dir)
        self.dirs['validation_converted'] = os.path.join(self.dirs['validation'], validation_converted_dir)
        self.dirs['validation_log'] = os.path.join(self.dirs['validation'], validation_log_dir)
        os.makedirs(self.dirs['validation'], exist_ok=True)
        os.makedirs(self.dirs['validation_result'], exist_ok=True)

        # 7] Test
        self.dirs['test_data'] = test_data_dir
        self.dirs['test_pathlist'] = test_pathlist_dir
        self.dirs['test'] = os.path.join(self.dirs['exp'], test_dir)
        self.dirs['test_result'] = os.path.join(self.dirs['test'], test_result_dir)
        self.dirs['test_converted'] = os.path.join(self.dirs['test'], test_converted_dir)
        os.makedirs(self.dirs['test'], exist_ok=True)
        os.makedirs(self.dirs['test_result'], exist_ok=True)
        os.makedirs(self.dirs['test_converted'], exist_ok=True)

    def build_model(self, params = None):
        '''Initialize NN model & optimizers & lr_schedulers
        '''
        # 1] Models
        self.Encoder = cc(Encoder(label_num = self.num_speakers))
        self.Decoder = [cc(Decoder()) for i in range(self.num_speakers)]
        self.SC = cc(SpeakerClassifier(label_num = self.num_speakers))
        self.ASR = cc(AutomaticSpeechRecognizer())
        self.AC = cc(AuxiliaryClassifier(label_num = self.num_speakers))
        # 2] Optimizers
        decoder_parameter_list = list()
        for decoder in self.Decoder:
            decoder_parameter_list += list(decoder.parameters())
        vae_params = list(self.Encoder.parameters()) + decoder_parameter_list
        self.optimizer = dict()
        self.optimizer['VAE'] = optim.Adam(vae_params, lr=params['vae_lr'], betas=params['vae_betas'])
        self.optimizer['SC'] = optim.Adam(self.SC.parameters(), lr=params['sc_lr'], betas=params['sc_betas'])
        self.optimizer['ASR'] = optim.Adam(self.ASR.parameters(), lr=params['asr_lr'], betas=params['asr_betas'])
        self.optimizer['AC'] = optim.Adam(self.AC.parameters(), lr=params['ac_lr'], betas=params['ac_betas'])

        # 3] lr_schedulers
        def lr_schedule_VAE(epoch):
            lr_target = 1e-3
            lr_decay = (1 - lr_target) / 200
            return 1 - lr_decay * epoch
        lr_schedule = dict(
        VAE = lr_schedule_VAE
        )
        self.lr_scheduler = dict()
        # self.lr_scheduler['VAE'] = optim.lr_scheduler.MultiStepLR(self.optimizer['VAE'], milestones=[50, 100], gamma=0.1)
        # self.lr_scheduler['VAE'] = optim.lr_scheduler.LambdaLR(self.optimizer['VAE'], lr_lambda=lr_schedule['VAE'])
        self.lr_scheduler['VAE'] = optim.lr_scheduler.ExponentialLR(self.optimizer['VAE'], gamma=(1e-2) ** (1/200))
        self.lr_scheduler['SC'] = None
        self.lr_scheduler['ASR'] = None
        self.lr_scheduler['AC'] = None

    def set_train_param(self, train_param):
        for param, value in train_param.items():
            self.train_p[param] = value
        assert self.train_p['iter_per_ep'] * self.train_p['mini_batch_size'] == self.train_p['batch_size'], 'Specified batch_size "%s" cannot be divided by mini_batch_size "%s"'%(self.train_p['batch_size'], self.train_p['mini_batch_size'])

    def save_model(self, model_path, epoch):
        all_model=dict()
        all_model['Encoder'] = self.Encoder.state_dict()
        for i, decoder in enumerate(self.Decoder):
            module_name = 'Decoder_' + str(i)
            all_model[module_name] = decoder.state_dict()

        all_model['SpeakerClassifier'] = self.SC.state_dict()
        all_model['AutomaticSpeechRecognizer'] = self.ASR.state_dict()
        all_model['AuxiliaryClassifier'] = self.AC.state_dict()
        all_model['optimizer_VAE'] = self.optimizer['VAE'].state_dict()
        all_model['optimizer_SC'] = self.optimizer['SC'].state_dict()
        all_model['optimizer_ASR'] = self.optimizer['ASR'].state_dict()
        all_model['optimizer_AC'] = self.optimizer['AC'].state_dict()

        new_model_path = os.path.join(model_path,'epoch_{}.pt'.format(epoch))
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path, epoch, speaker_num):
        speaker_num = int(speaker_num)
        model_path = os.path.join(model_path,'epoch_{}.pt'.format(epoch))
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['Encoder'])
            decoder_name = 'Decoder_'+str(speaker_num)
            self.Decoder[speaker_num].load_state_dict(all_model[decoder_name])

    def load_model_vae(self, model_path, epoch):
        model_path = os.path.join(model_path,'epoch_{}.pt'.format(epoch))
        print('load VAE model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['Encoder'])
            for speaker_num in range(self.num_speakers):
                self.Decoder[speaker_num].load_state_dict(all_model['Decoder_'+str(speaker_num)])

    def load_model_all(self, model_path, epoch):
        model_path = os.path.join(model_path,'epoch_{}.pt'.format(epoch))
        print('load ALL model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['Encoder'])
            for speaker_num in range(self.num_speakers):
                self.Decoder[speaker_num].load_state_dict(all_model['Decoder_'+str(speaker_num)])
            self.SC.load_state_dict(all_model['SpeakerClassifier'])
            self.ASR.load_state_dict(all_model['AutomaticSpeechRecognizer'])
            self.AC.load_state_dict(all_model['AuxiliaryClassifier'])
            self.optimizer['VAE'].load_state_dict(all_model['optimizer_VAE'])
            self.optimizer['SC'].load_state_dict(all_model['optimizer_SC'])
            self.optimizer['ASR'].load_state_dict(all_model['optimizer_ASR'])
            self.optimizer['AC'].load_state_dict(all_model['optimizer_AC'])

    def set_train(self):
        self.Encoder.train()
        for decoder in self.Decoder:
            decoder.train()

    def set_eval(self,trg_speaker_num):
        self.Encoder.eval()
        self.Decoder[trg_speaker_num].eval()

    def grad_reset(self):
        self.optimizer['SC'].zero_grad()
        self.optimizer['ASR'].zero_grad()
        self.optimizer['AC'].zero_grad()
        self.optimizer['VAE'].zero_grad()

    def generate_label(self, index, batch_size):
        '''label == [index, index, ...] with len(label) == batch_size '''
        labels = [index] * batch_size
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
            mu_z, logvar_z = self.Encoder(x,c_src)
            mu_xt, logvar_xt = self.Decoder[trg_speaker_num](mu_z)
        return mu_xt.detach().cpu().numpy()

    def loss_MDVAE(self, x, c_onehot):
        '''Get KLD loss & Reconstruction loss(negative gaussian log likelihood)
        x_src - [Encoder] - z (- KLD loss) - [Decoder(All)] - xt (- Rec loss)
        For: VAE
        '''
        mu_z, logvar_z = self.Encoder(x, c_onehot)
        loss_KLD = KLD_loss(mu_z, logvar_z)
        z = self.reparameterize(mu_z, logvar_z)
        # Decoder - All
        loss_rec_list = list()
        for i in range(self.num_speakers):
            mu_xt, logvar_xt = self.Decoder[i](z)
            loss_rec = -GaussianLogDensity(x, mu_xt, logvar_xt) # Maximize probability
            loss_rec_list.append(loss_rec)
        loss_rec = sum(loss_rec_list) / self.num_speakers # Mean loss
        return loss_KLD, loss_rec

    def loss_SI(self, x, c_onehot, c):
        '''Get Speaker Classifier loss
        x_src - [Encoder] - z - [Speaker Classifier] - logits - loss
        For: SC, VAE
        '''
        mu, logvar = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.SC(z)
        loss_SI = F.cross_entropy(logits, c)
        return loss_SI

    def loss_LI(self, x, c_onehot, si):
        '''Get Automatic Speech Recognizer loss
        x_src - [Encoder] - z - [Automatic Speech Recognizer] - logits - loss
        For: ASR, VAE
        '''
        mu, logvar = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        logits = self.ASR(z)
        loss_LI = F.cross_entropy(logits, si)
        return loss_LI

    def loss_ACforAC(self, x, c):
        '''Get Auxiliary Classifier loss for AC
        x - [Auxiliary Classifier] - logits - loss
        For: AC
        '''
        logits = self.AC(x)
        loss_AC = F.cross_entropy(logits, c)
        return loss_AC

    def loss_ACforVAE(self, x, c_onehot):
        '''Get Auxiliary Classifier loss for VAE
        x - [Encoder] - z - [Decoder(All)] - xt - [Auxiliary Classifier] - logits - loss
        For: VAE
        '''
        mu_z, logvar_z = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu_z, logvar_z)
        # Decoder - All
        loss_AC_list = list()
        for i in range(self.num_speakers):
            mu_xt, logvar_xt = self.Decoder[i](z)
            xt = self.reparameterize(mu_xt, logvar_xt)
            logits = self.AC(xt)
            c = self.generate_label(i, self.train_p['mini_batch_size'])
            loss_AC = F.cross_entropy(logits, c)
            loss_AC_list.append(loss_AC)
        loss_AC = sum(loss_AC_list) / self.num_speakers # Mean loss
        return loss_AC

    def loss_SC(self, x, c_onehot):
        '''Get Semantic Consistency loss
        x - [Encoder] - z - [Decoder(All)] - xt - [Encoder] - zt - loss
        For: VAE
        '''
        mu_z, logvar_z = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu_z, logvar_z)
        # Decoder - All
        loss_SC_list = list()
        for i in range(self.num_speakers):
            mu_xt, logvar_xt = self.Decoder[i](z)
            xt = self.reparameterize(mu_xt, logvar_xt)
            ct = self.generate_label(i, self.train_p['mini_batch_size'])
            ct_onehot = self.label2onehot(ct)
            mu_zt, logvar_zt = self.Encoder(xt, ct_onehot)
            zt = self.reparameterize(mu_zt, logvar_zt)
            loss_SC = (z - zt) ** 2
            loss_SC_list.append(loss_SC)
        loss_SC = sum(loss_SC_list) / self.num_speakers
        return loss_SC

    def loss_C(self, x, c_onehot, index):
        '''Get Cycle loss
        x - [Encoder] - z - [Decoder(All)] - xt - [Encoder] - zt (- KLD loss) - [Decoder(? original?)] - xtt (- Rec loss)
        For: VAE
        '''
        mu_z, logvar_z = self.Encoder(x, c_onehot)
        z = self.reparameterize(mu_z, logvar_z)
        # Decoder - All
        loss_C_KLD_list = list()
        loss_C_rec_list = list()
        for i in range(self.num_speakers):
            mu_xt, logvar_xt = self.Decoder[i](z)
            xt = self.reparameterize(mu_xt, logvar_xt)
            ct = self.generate_label(i, self.train_p['mini_batch_size'])
            ct_onehot = self.label2onehot(ct)
            mu_zt, logvar_zt = self.Encoder(xt, ct_onehot)
            loss_C_KLD = KLD_loss(mu_zt, logvar_zt)
            loss_C_KLD_list.append(loss_C_KLD)

            zt = self.reparameterize(mu_zt, logvar_zt)
            mu_xtt, logvar_xtt = self.Decoder[index](zt)
            loss_C_rec_list.append(loss_C_rec)
        loss_C_KLD = sum(loss_C_KLD_list) / self.num_speakers
        loss_C_rec = sum(loss_C_rec_list) / self.num_speakers
        return loss_C_KLD, loss_C_rec

    def train(self, lambd = None, lambda_norm = False, train_param = None, train_data_dir = None, model_dir = None):
        # 0] Manual Directory&Parmeter designation
        if lambd is None:
            lambd = self.lambd
        if train_data_dir is not None:
            self.dirs['train_data'] = train_data_dir
        if model_dir is not None:
            self.dirs['model'] = model_dir
        if train_param is not None:
            self.set_train_param(train_param)

        # 1] Loss Normalization value
        lambd_total = sum(lambd.values()) + 1

        # 2] Validation Settings
        loss_index = ['VAE','MDVAE','SI','LI','AC','Sem','Cycle']
        loss_summary = pd.DataFrame(columns = loss_index)
        validation_summary = pd.DataFrame(columns = self.p_measure_list)
        self.dirs['validation_summary'] = os.path.join(self.dirs['validation'], 'validation_summary.p')

        # 3] Print Train Settings
        # self.p == Printer prints out to stdout & log
        self.p.add('Current Time:'+time.strftime('%Y-%m-%d %H:%M %Ss')+'\n')
        self.p.add('Experiment run on: %s\n'%platform.node())
        self.p.add('Start Training with the following setting:\n')
        self.p.add(stars()+'\n')
        self.p.add('torch version: %s\n'%torch.__version__)
        self.p.add('train_data_dir: %s\n'%self.dirs['train_data'])
        self.p.add('model_dir: %s\n'%self.dirs['model'])
        self.p.add('num_speakers: %s\n'%self.num_speakers)
        self.p.add('training parameters: %s\n'%self.train_p)
        self.p.add('lambd: %s\n'%lambd)
        self.p.add('lambd_total: %s\n'%lambd_total)
        self.p.add('lambda normalization: %s\n'%lambda_norm)
        self.p.add('model optimizer configurations:\n')
        self.p.add('VAE: %s\n'%self.optimizer['VAE'].defaults)
        self.p.add('SC: %s\n'%self.optimizer['SC'].defaults)
        self.p.add('ASR: %s\n'%self.optimizer['ASR'].defaults)
        self.p.add('AC: %s\n'%self.optimizer['AC'].defaults)
        self.p.add('model lr_scheduler configurations:\n')
        for module, scheduler in self.lr_scheduler.items():
            if scheduler is not None:
                self.p.add(module+': %s\n'%scheduler.state_dict())
        self.p.add(stars()+'\n')
        self.p.print(end='')

        append(self.dirs['loss_log'], 'epoch '+' '.join(loss_index)+'\n')
        append(self.dirs['validation_log'], 'epoch '+' '.join(self.p_measure_list)+'\n')
        loss_SI = None
        loss_LI = None
        loss_AC = None
        loss_SC = None
        loss_C = None
        np.random.seed(0)
        # 4] Start training
        for ep in range(self.train_p['start_epoch'], self.train_p['start_epoch'] + self.train_p['n_epoch']):
            self.p.print('Epoch:%s'%ep)
            self.p.print('VAE lr:%s'%self.optimizer['VAE'].param_groups[0]['lr'])
            # Keep track of loss results for monitoring
            loss_list = list()
            time_start = time.time()
            for i, src_speaker in enumerate(self.speaker_list):
                for j, trg_speaker in enumerate(self.speaker_list):

                    # Load train data: mcep, state index
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
                    # siset_A = np.expand_dims(siset_A, axis=1)
                    # siset_B = np.expand_dims(siset_B, axis=1)
                    c_A = self.generate_label(i, self.train_p['mini_batch_size']).to(self.device, dtype = torch.long)
                    c_B = self.generate_label(j, self.train_p['mini_batch_size']).to(self.device, dtype = torch.long)
                    c_onehot_A = self.label2onehot(c_A).to(self.device, dtype = torch.float)
                    c_onehot_B = self.label2onehot(c_B).to(self.device, dtype = torch.float)

                    for iteration in range(self.train_p['iter_per_ep']):
                        start = iteration * self.train_p['mini_batch_size']
                        end = (iteration + 1) * self.train_p['mini_batch_size']

                        x_batch_A = torch.as_tensor(dataset_A[start:end], device = self.device, dtype=torch.float)
                        x_batch_B = torch.as_tensor(dataset_B[start:end], device = self.device, dtype=torch.float)
                        si_batch_A = torch.as_tensor(siset_A[start:end], device = self.device, dtype=torch.long)
                        si_batch_B = torch.as_tensor(siset_B[start:end], device = self.device, dtype=torch.long)

                        self.grad_reset()
                        loss_VAE = 0
                        '''It's okay to assign same variable name to different graph tensors.'''
                        if lambd['SI'] != 0:
                            # 1. Update SpeakerClssifier (SC) module
                            loss_SI_A = self.loss_SI(x_batch_A, c_onehot_A, c_A)
                            loss_SI_B = self.loss_SI(x_batch_B, c_onehot_B, c_B)
                            loss_SI = loss_SI_A + loss_SI_B # Loss_SI used for SC
                            self.optimizer['SC'].zero_grad()
                            loss_SI.backward()
                            self.optimizer['SC'].step()
                            del loss_SI, loss_SI_A, loss_SI_B # Delete to free memory
                            # 2. Get loss_SI for VAE
                            loss_SI_A = self.loss_SI(x_batch_A, c_onehot_A, c_A)
                            loss_SI_B = self.loss_SI(x_batch_B, c_onehot_B, c_B)
                            loss_SI = -(loss_SI_A + loss_SI_B) # Loss_SI used for VAE
                            loss_VAE += lambd['SI'] * loss_SI

                        if lambd['LI'] != 0:
                            # 1. Update AutomaticSpeechRecognizer (ASR) module
                            loss_LI_A = self.loss_LI(x_batch_A, c_onehot_A, si_batch_A)
                            loss_LI_B = self.loss_LI(x_batch_B, c_onehot_B, si_batch_B)
                            loss_LI = loss_LI_A + loss_LI_B
                            self.optimizer['ASR'].zero_grad()
                            loss_LI.backward()
                            self.optimizer['ASR'].step()
                            del loss_LI, loss_LI_A, loss_LI_B
                            # 2. Get loss_LI for VAE
                            loss_LI_A = self.loss_LI(x_batch_A, c_onehot_A, si_batch_A)
                            loss_LI_B = self.loss_LI(x_batch_B, c_onehot_B, si_batch_B)
                            loss_LI = loss_LI_A + loss_LI_B
                            loss_VAE += lambd['LI'] * loss_LI

                        if lambd['AC'] != 0:
                            # 1. Update AuxiliaryClassifier (AC) module
                            loss_AC_A = self.loss_ACforAC(x_batch_A, c_A)
                            loss_AC_B = self.loss_ACforAC(x_batch_B, c_B)
                            loss_AC = loss_AC_A + loss_AC_B
                            self.optimizer['AC'].zero_grad()
                            loss_AC.backward()
                            self.optimizer['AC'].step()
                            # 2. Get loss_LI for VAE
                            loss_AC_A = self.loss_ACforAC(x_batch_A, c_A)
                            loss_AC_B = self.loss_ACforAC(x_batch_B, c_B)
                            loss_AC = loss_AC_A + loss_AC_B
                            loss_VAE += lambd['AC'] * loss_AC

                        if lambd['SC'] != 0:
                            # 1. Get Loss_SC for VAE
                            loss_SC_A = self.loss_SC(x_batch_A, c_onehot_A)
                            loss_SC_B = self.loss_SC(x_batch_B, c_onehot_B)
                            loss_SC = loss_SC_A + loss_SC_B
                            loss_VAE += lambd['SC'] * loss_SC

                        if lambd['C'] != 0:
                            # 1. Get Loss_C for VAE
                            loss_C_A = self.loss_C(x_batch_A, c_onehot_A, i)
                            loss_C_B = self.locc_C(x_batch_B, c_onehot_B, j)
                            loss_C = loss_C_A + loss_C_B
                            loss_VAE += lambd['C'] * loss_C

                        loss_KLD_A, loss_rec_A = self.loss_MDVAE(x_batch_A, c_onehot_A)
                        loss_KLD_B, loss_rec_B = self.loss_MDVAE(x_batch_B, c_onehot_B)
                        loss_MDVAE = loss_KLD_A + loss_rec_A + loss_KLD_B + loss_rec_B
                        loss_VAE += loss_MDVAE
                        if lambda_norm == True:
                            loss_VAE /= lambd_total # Normalize so that the lambda will effect the learning rate less
                        self.optimizer['VAE'].zero_grad()
                        loss_VAE.backward()
                        self.optimizer['VAE'].step()

                        loss_list.append([float(loss_VAE), float(loss_MDVAE), loss_SI, loss_LI, loss_AC, loss_SC, loss_C])
            self.lr_scheduler['VAE'].step()
            time_end = time.time()
            time_elapsed = time_end - time_start

            # 1. Loss Results
            loss_result = pd.DataFrame(loss_list, columns = loss_index)
            loss_result_mean = loss_result.mean()
            # 2. Save results
            loss_log_content = str(ep)
            for value in loss_result_mean:
                loss_log_content += ' '+str(value)
            append(self.dirs['loss_log'], loss_log_content+'\n')
            loss_summary.loc[ep] = loss_result_mean.values
            # 3. Print
            self.p.print('Time elapsed this epoch: {:0.1f}m {:0.5}s'.format( time_elapsed // 60, time_elapsed % 60 ))
            self.p.print('Mean\n'+ str(loss_result.mean().to_frame().T))
            self.p.print('Std\n' + str(loss_result.std().to_frame().T))
            # 4. PLot
            for loss in loss_summary.columns:
                fig_save_dir = os.path.join(self.dirs['exp'], 'loss_'+loss+'.png')
                axes = loss_summary.plot(y = loss, style='o-')
                fig = axes.get_figure()
                fig.savefig(fig_save_dir)
            plt.close('all')

            # Save model (Default: 1)
            if (ep) % self.train_p['model_save_epoch'] == 0:
                self.p.print("Saving Model, epoch: {}".format(ep))
                self.save_model(self.dirs['model'], ep)

            # Validation (Default: 1)
            if (ep) % self.train_p['validation_epoch'] == 0:
                '''validation_result : holds validation result of all samples at particular epoch
                validation_summary: holds mean value of each validation results
                '''
                self.p.print('-'*50)
                self.p.print('Validation Start.')
                # 1. Get performance measures
                validation_result = self.performance_measure(test_data_dir = self.dirs['validation_data'], test_pathlist_dir = self.dirs['validation_pathlist'], sample_per_path = self.train_p['sample_per_path'])
                validation_result_mean = validation_result.mean()
                # 2. Save results
                validation_summary.loc[ep] = validation_result_mean.values
                save_pickle(validation_summary, self.dirs['validation_summary'])

                validation_log_content = str(ep)
                for value in validation_result_mean:
                    validation_log_content += ' '+str(value)
                append(self.dirs['validation_log'], validation_log_content+'\n')
                # 3. Print
                self.p.print("Validation Complete.")
                self.p.print('Mean\n' + str(validation_result.mean().to_frame().T))
                self.p.print('Std\n' + str(validation_result.std().to_frame().T))
                # 4. Plot
                for measure in validation_summary.columns:
                    fig_save_dir = os.path.join(self.dirs['validation'], measure+'.png')
                    axes = validation_summary.plot(y = measure, style='o-')
                    fig = axes.get_figure()
                    fig.savefig(fig_save_dir)
                plt.close('all')
                self.set_train()
                self.p.print('-'*50)
        # After training ends, convert wav with the best model
        best_ep = validation_summary['mcd'].idxmin()
        self.load_model_vae(self.dirs['model'], best_ep)
        self.convert(test_data_dir = self.dirs['validation_data'], test_pathlist_dir = self.dirs['validation_pathlist_dir'], test_converted_dir = self.dirs['validation_converted'])

    def performance_measure(self, test_data_dir = None, test_pathlist_dir = None, sample_per_path = 10):
        # Default to validation directories, unless there's additional information
        if test_data_dir is  None:
            test_data_dir = self.dirs['test_data']
        if test_pathlist_dir is None:
            test_pathlist_dir = self.dirs['test_pathlist']

        test_pathlist = read(test_pathlist_dir).splitlines()

        # 1] Convert all data
        print('Loading&Converting test data...')
        converted_mcep_list = list()
        target_mcep_list = list()
        tested_pathlist = list()
        start_time = time.time()
        # np.random.seed(0)
        # np.random.shuffle(test_pathlist)
        for speaker_A in self.speaker_list:
            for speaker_B in self.speaker_list:
                n_sample = 0
                if speaker_A == speaker_B:
                    continue
                for test_path in test_pathlist:
                    if n_sample >= sample_per_path:
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
                    data_A_dir = os.path.join(test_data_dir, src_speaker, '{}.p'.format(filename_src))
                    data_B_dir = os.path.join(test_data_dir, trg_speaker, '{}.p'.format(filename_trg))
                    train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
                    train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))

                    # Load data
                    _, _, coded_sps_mean_A, coded_sps_std_A, _, _ = load_pickle(train_data_A_dir)
                    _, _, coded_sps_mean_B, coded_sps_std_B, _, _ = load_pickle(train_data_B_dir)
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
        n_sample = len(tested_pathlist)
        pool = Pool(30)
        converted_ms_list = pool.map(extract_ms, converted_mcep_list)
        target_ms_list = pool.map(extract_ms, target_mcep_list)
        pool.close()
        pool.join()

        pool = Pool(30)
        mcd_list = pool.starmap(mcd_cal, zip(converted_mcep_list, target_mcep_list))
        msd_all_list = pool.starmap(msd_cal, zip(converted_ms_list, target_ms_list, ['all']*n_sample))
        msd_vector_list = pool.starmap(msd_cal, zip(converted_ms_list, target_ms_list, ['vector']*n_sample))
        gv_list = pool.starmap(gv_cal, zip(converted_mcep_list))
        pool.close()
        pool.join()

        # 3] Save Results
        test_result = pd.DataFrame(index = tested_pathlist, columns = self.p_measure_list, dtype = float)
        print('Calculation complete.')
        for test_path, mcd, msd_all, msd_vector, gv in zip(tested_pathlist, mcd_list, msd_all_list, msd_vector_list, gv_list):
            test_result.loc[test_path] = mcd, msd_all, msd_vector, gv

        end_time = time.time()
        time_elapsed = end_time - start_time
        self.p.print('Time elapsed for testing: {:0.1f}m {:0.5}s'.format( time_elapsed // 60, time_elapsed % 60 ))
        return test_result

    def convert(self, test_data_dir = None, test_pathlist_dir = None, test_converted_dir = None, sample_per_path = 10):
        # 0] Default to test directories, unless there's additional information
        if test_data_dir is  None:
            test_data_dir = self.dirs['test_data']
        if test_pathlist_dir is None:
            test_pathlist_dir = self.dirs['test_pathlist']
        if test_converted_dir is None:
            test_converted_dir = self.dirs['test_converted']
        test_pathlist = read(test_pathlist_dir).splitlines()

        # 1] Convert all data & Save them
        print('Loading&Converting test data...')
        start_time = time.time()
        np.random.seed(0)
        np.random.shuffle(test_pathlist)
        for speaker_A in self.speaker_list:
            for speaker_B in self.speaker_list:
                n_sample = 0
                if speaker_A == speaker_B:
                    continue
                for test_path in test_pathlist:
                    if n_sample >= sample_per_path:
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
                    data_A_dir = os.path.join(test_data_dir, src_speaker, '{}.p'.format(filename_src))
                    data_B_dir = os.path.join(test_data_dir, trg_speaker, '{}.p'.format(filename_trg))
                    train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
                    train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))

                    # Load data
                    _, _, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A = load_pickle(train_data_A_dir)
                    _, _, coded_sps_mean_B, coded_sps_std_B, log_f0s_mean_B, log_f0s_std_B = load_pickle(train_data_B_dir)
                    coded_sp, ap, f0 = load_pickle(data_A_dir) # coded_sp.shape: (T, 36)

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
                    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A, mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)

                    coded_sp_converted = coded_sp_converted_norm * coded_sps_std_B + coded_sps_mean_B
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                    decoded_sp_converted = world_decode_mc(mc=coded_sp_converted, fs=self.preprocess_p['sr'])
                    if coded_sp_converted_norm.shape[1] < len(f0):
                        f0_converted = f0_converted[:int(coded_sp_converted_norm.shape[1])]
                        ap = ap[:int(coded_sp_converted_norm.shape[1])]
                        self.p.print('shape not coherent?? file:%s'%(filename_src))
                        self.p.print(test_path)
                    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=self.preprocess_p['sr'], frame_period=self.preprocess_p['frame_period'])
                    wav_transformed = np.nan_to_num(wav_transformed)
                    # Save file
                    soundfile.write(os.path.join(test_converted_dir, test_path+'.wav'), data=wav_transformed, samplerate=self.preprocess_p['sr'])
        print('Converted files saved.')
        end_time = time.time()
        time_elapsed = end_time - start_time
        self.p.print('Time elapsed for converting: {:0.1f}m {:0.5}s'.format( time_elapsed // 60, time_elapsed % 60 ))

    def test(self, test_data_dir = None, test_pathlist_dir = None, test_converted_dir = None, sample_per_path = 10):
        '''Calculate Performance measures & Save converted files
        This is same with calling
        performance_measure() and convert()
        but is more efficient
        '''
        # 0] Default to test directories, unless there's additional information
        if test_data_dir is  None:
            test_data_dir = self.dirs['test_data']
        if test_pathlist_dir is None:
            test_pathlist_dir = self.dirs['test_pathlist']
        if test_converted_dir is None:
            test_converted_dir = self.dirs['test_converted']

        test_pathlist = read(test_pathlist_dir).splitlines()

        # 1] Convert all data & Save them
        print('Loading&Converting test data...')
        converted_mcep_list = list()
        target_mcep_list = list()
        tested_pathlist = list()
        sample_per_path = 1
        start_time = time.time()
        np.random.seed(0)
        np.random.shuffle(test_pathlist)
        for speaker_A in self.speaker_list:
            for speaker_B in self.speaker_list:
                n_sample = 0
                if speaker_A == speaker_B:
                    continue
                for test_path in test_pathlist:
                    if n_sample >= sample_per_path:
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
                    data_A_dir = os.path.join(test_data_dir, src_speaker, '{}.p'.format(filename_src))
                    data_B_dir = os.path.join(test_data_dir, trg_speaker, '{}.p'.format(filename_trg))
                    train_data_A_dir = os.path.join(self.dirs['train_data'], src_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))
                    train_data_B_dir = os.path.join(self.dirs['train_data'], trg_speaker, 'cache{}.p'.format(self.preprocess_p['num_mcep']))

                    # Load data
                    _, _, coded_sps_mean_A, coded_sps_std_A, _, _ = load_pickle(train_data_A_dir)
                    _, _, coded_sps_mean_B, coded_sps_std_B, _, _ = load_pickle(train_data_B_dir)
                    coded_sp, ap, f0 = load_pickle(data_A_dir) # coded_sp.shape: (T, 36)
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
                    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A, mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)

                    coded_sp_converted = coded_sp_converted_norm * coded_sps_std_B + coded_sps_mean_B
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                    decoded_sp_converted = world_decode_mc(mc=coded_sp_converted, fs=self.preprocess_p['sr'])
                    if coded_sp_converted_norm.shape[1] < len(f0):
                        f0_converted = f0_converted[:int(coded_sp_converted_norm.shape[1])]
                        ap = ap[:int(coded_sp_converted_norm.shape[1])]
                        self.p.print('shape not coherent?? file:%s'%(filename_src))
                        self.p.print(test_path)
                    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=self.preprocess_p['sr'], frame_period=self.preprocess_p['frame_period'])
                    wav_transformed = np.nan_to_num(wav_transformed)
                    # Save file
                    librosa.output.write_wav(os.path.join(test_converted_dir, test_path+'.wav'), wav_transformed, self.preprocess_p['sr'])

                    converted_mcep_list.append(coded_sp_converted)
                    target_mcep_list.append(coded_sp_trg)
                    tested_pathlist.append(test_path)
        print('Converted files saved.')

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

if False:
    # self = Experiment(num_speakers = 4
    # Dummy class for debugging
    print('DEBUGGING!!')
    # class Dummy():
    #     """ Dummy class for debugging """
    #     def __init__(self):
    #         pass
    # self = Dummy()
    self = Experiment(num_speakers = 4)
    self.dirs = dict()
    exp_dir = 'exp/'
    model_dir = 'model/'
    log_dir = 'log.txt'
    train_data_dir = 'processed/'
    si_dir = 'processed_stateindex/'
    validation_dir = 'validation/'
    validation_result_dir = 'validation_result/'
    validation_data_dir = 'processed_validation/inset_dev/'
    validation_pathlist_dir = 'filelist/inset_dev.lst'
    validation_log_dir = 'validation_log.txt'
    test_dir = 'test/'
    test_data_dir = 'processed_validation/inset_test/'
    test_pathlist_dir = 'filelist/inset_test.lst'
    self.dirs['exp'] = exp_dir
    model_dir = os.path.join(self.dirs['exp'], model_dir)
    self.dirs['model'] = model_dir
    log_dir = os.path.join(self.dirs['exp'], log_dir)
    self.dirs['log'] = log_dir
    self.dirs['train_data'] = train_data_dir
    self.dirs['si'] = si_dir
    self.dirs['validation_data'] = validation_data_dir
    self.dirs['validation_pathlist'] = validation_pathlist_dir
    validation_dir = os.path.join(self.dirs['exp'], validation_dir)
    self.dirs['validation'] = validation_dir
    validation_result_dir = os.path.join(self.dirs['validation'], validation_result_dir)
    self.dirs['validation_result'] = validation_result_dir
    validation_log_dir = os.path.join(self.dirs['validation'], validation_log_dir)
    self.dirs['validation_log'] = validation_log_dir

    self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
    self.num_speakers = len(self.speaker_list)
    self.train_p = dict()
    self.train_p['n_train_frames'] = 128
    self.train_p['iter_per_ep'] = 4
    self.train_p['start_epoch'] = 1
    self.train_p['n_epoch'] = 200
    self.train_p['batch_size'] = 8
    self.train_p['model_save_epoch'] = 2
    self.train_p['validation_epoch'] = 2
    self.train_p['sample_per_path'] = 10
    self.preprocess_p = dict(
    sr = 16000,
    frame_period = 5.0,
    num_mcep = 36,
    )
    test_pathlist = read(self.dirs['validation_pathlist']).splitlines()
    test_path = test_pathlist[0]
    i=0
    j=2
    src_speaker = self.speaker_list[i]
    trg_speaker = self.speaker_list[j]
    iteration=0
    torch.manual_seed(0)

    loss_index = ['VAE','MDVAE','SI','LI','AC','Sem','Cycle']
    loss_summary = pd.DataFrame(columns = loss_index)
    validation_summary = pd.DataFrame(columns = self.p_measure_list)
# x=torch.arange(2*3*4).view(2,3,1,4)
# print(x)
# print(torch.flip(x, [0,1]))
# torch.flip(x,[0,1]).shape
# x.transpose(0,1).shape
# x.transpose(0,1)[1,0,2]
# x.squeeze()
# x.shape
# x[0,1,2]
# x=torch.ones((8,8,1,32))
# x.shape
# y = self.ASR(x)
# y.shape
# y.transpose(1,2).shape
# F.cross_entropy(y.transpose(1,2), si_batch_A)
# c_A.shape
# y[0].shape
# for s, yy in zip(si_batch_A, y):
#     l+=F.cross_entropy(yy,s)
#
# l/8
# l=torch.tensor(0).to(float)
# si_batch_A[0].shape
# y.transpose(1,2)[0].shape
# F.cross_entropy(y, si_batch_A)
# help(F.cross_entropy)
# si_batch_A.shape
# iteration=0
