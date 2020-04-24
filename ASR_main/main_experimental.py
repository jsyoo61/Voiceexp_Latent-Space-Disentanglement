import argparse
import torch
from experiment import Experiment
from tools.tools import str2bool

if __name__ == '__main__':
    # Default Hyperparameters
    SI = 0
    LI = 0
    AC = 0
    SC = 0
    C = 0
    vae_lr = 0.01
    vae_betas = (0.9,0.999)
    sc_lr = 0.0002
    sc_betas = (0.5,0.999)
    asr_lr = 0.00001
    asr_betas = (0.5,0.999)
    ac_lr = 0.00005
    ac_betas = (0.5,0.999)

    parser = argparse.ArgumentParser(description = 'Proceed experiment with specified exp_name')
    parser.add_argument('--exp_name', type = str, help = 'Experiment name. All files will be stored in exp/exp_name')
    parser.add_argument('--new', default = True, type = str2bool, help = 'if True, create new model')
    # Loss lambda
    parser.add_argument('--SI', default = SI, type = float, help = 'lambda_SI')
    parser.add_argument('--LI', default = LI, type = float, help = 'lambda_LI')
    parser.add_argument('--AC', default = AC, type = float, help = 'lambda_AC')
    parser.add_argument('--SC', default = SC, type = float, help = 'lambda_SC')
    parser.add_argument('--C', default = C, type = float, help = 'lambda_C')
    parser.add_argument('--lambda_norm', default = True, type = str2bool, help = 'if True, normalize loss with sum(lambda)')
    # Learning Rate
    parser.add_argument('--vae_lr', default = vae_lr, type = float, help = 'vae_lr')
    parser.add_argument('--vae_betas', default = vae_betas, type = float, help = 'vae_betas')
    parser.add_argument('--sc_lr', default = sc_lr, type = float, help = 'sc_lr')
    parser.add_argument('--sc_betas', default = sc_betas, type = float, help = 'sc_betas')
    parser.add_argument('--asr_lr', default = asr_lr, type = float, help = 'asr_lr')
    parser.add_argument('--asr_betas', default = asr_betas, type = float, help = 'asr_betas')
    parser.add_argument('--ac_lr', default = ac_lr, type = float, help = 'ac_lr')
    parser.add_argument('--ac_betas', default = ac_betas, type = float, help = 'ac_betas')
    # Training parameters
    parser.add_argument('--n_epoch', type = int, help = 'number of training epochs')
    parser.add_argument('--batch_size', type = int, help = 'number of batch size')
    parser.add_argument('--model_save_epoch', type = int, help = 'model save interval')
    parser.add_argument('--validation_epoch', type = int, help = 'validation interval')
    parser.add_argument('-t','--train_data_dir')
    args = parser.parse_args()

    lambd = dict(
    SI = args.SI,
    LI = args.LI,
    AC = args.AC,
    SC = args.SC,
    C = args.C,
    )
    model_p = dict(
    vae_lr = args.vae_lr,
    vae_betas = args.vae_betas,
    sc_lr = args.sc_lr,
    sc_betas = args.sc_betas,
    asr_lr = args.asr_lr,
    asr_betas = args.asr_betas,
    ac_lr = args.ac_lr,
    ac_betas = args.ac_betas,
    )
    train_p_ = dict(
    n_epoch = args.n_epoch,
    batch_size = args.batch_size,
    model_save_epoch = args.model_save_epoch,
    validation_epoch = args.validation_epoch,
    )
    train_p = dict()
    for p in train_p_:
        if train_p_[p] is not None:
            train_p[p] = train_p_[p]

    solver = Experiment(num_speakers = 4, exp_name = args.exp_name, model_p = model_p, new = args.new, train_p = train_p)
    # solver = Experiment(num_speakers = 100, exp_name = args.exp_name, model_p = model_p, new = args.new)
    solver.train(lambd = lambd, lambda_norm = args.lambda_norm, train_data_dir = args.train_data_dir)
    # solver.train(lambd = lambd, train_param = train_p, train_data_dir = args.train_data_dir)

from experiment import Experiment
train_p = dict(
n_epoch = 2,

)
lambd = dict(
SI=0,
LI=0,
AC=0,
SC=0,
C=0,
)
solver = Experiment(num_speakers = 4, train_p = train_p)
solver.train_p
solver.train(lambd=lambd)
solver.speaker_list
