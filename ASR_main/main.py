from solver_multi_decoder import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', default='./processed/')
    parser.add_argument('--model_path', default='VAE_models/VAE_models-50')
    parser.add_argument('--model_iter', default='0')

    args = parser.parse_args()

    print(torch.__version__)
    solver = Solver()

    solver.train(batch_size = 8, exp_dir = args.dataset_path, mode='ASR_TIMIT')

    # solver.train(batch_size = 8, exp_dir = args.dataset_path, mode='ASR_TIMIT_GAN')
