import argparse
import torch
from experiment import Experiment
from tools.tools import str2bool

if __name__ == '__main__':
    # Default Hyperparameters
    rec = 1
    SI = 0
    LI = 0
    AC = 0
    SC = 0
    C = 0
    vae_lr = 1e-3
    sc_lr = 0.0002
    asr_lr = 0.00001
    ac_lr = 0.00005

    parser = argparse.ArgumentParser(description = 'Proceed experiment with specified exp_name')
    parser.add_argument('--exp_name', type = str, help = 'Experiment name. All files will be stored in exp/exp_name')
    parser.add_argument('--new', default = True, type = str2bool, help = 'if True, create new model')
    parser.add_argument('--debug', default = False, type = str2bool, help = 'if True, redirect stdout to exp/exp_name/log_all.txt')
    # Loss lambda
    parser.add_argument('--KLD', type = float, help = 'lambda_KLD')
    parser.add_argument('--rec', type = float, help = 'lambda_rec')
    parser.add_argument('--SI', type = float, help = 'lambda_SI')
    parser.add_argument('--LI', type = float, help = 'lambda_LI')
    parser.add_argument('--AC', type = float, help = 'lambda_AC')
    parser.add_argument('--SC', type = float, help = 'lambda_SC')
    parser.add_argument('--C', type = float, help = 'lambda_C')
    parser.add_argument('--lambda_norm', default = True, type = str2bool, help = 'if True, normalize loss with sum(lambda)')
    # Learning Rate
    parser.add_argument('--vae_lr',type = float, help = 'vae_lr')
    parser.add_argument('--sc_lr', type = float, help = 'sc_lr')
    parser.add_argument('--asr_lr',type = float, help = 'asr_lr')
    parser.add_argument('--ac_lr', type = float, help = 'ac_lr')
    # Training parameters
    # parser.add_argument('--n_epoch', type = int, help = 'number of training epochs')
    # parser.add_argument('--batch_size', type = int, help = 'number of batch size')
    # parser.add_argument('--model_save_epoch', type = int, help = 'model save interval')
    # parser.add_argument('--validation_epoch', type = int, help = 'validation interval')
    args = parser.parse_args()
    print(args.__dict__)
    lambd_list = ['KLD', 'rec', 'SI', 'LI', 'AC', 'SC', 'C']
    lambd = {lambd_ : args.__dict__[lambd_] for lambd_ in lambd_list if args.__dict__[lambd_] is not None}
    print(lambd)

    # a= {1:None, 2:30}
    # b={l : a[l] for l in a.keys() if a}
    model_p = dict(
    vae_lr = args.vae_lr,
    sc_lr = args.sc_lr,
    asr_lr = args.asr_lr,
    ac_lr = args.ac_lr,
    )
    # train_p_ = dict(
    # n_epoch = args.n_epoch,
    # batch_size = args.batch_size,
    # model_save_epoch = args.model_save_epoch,
    # validation_epoch = args.validation_epoch,
    # )
    # train_p = dict()
    # for p in train_p_:
    #     if train_p_[p] is not None:
    #         train_p[p] = train_p_[p]
