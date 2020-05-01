import os
import shutil
import glob
import argparse
import time

def sort_results(sort_target, exp_dir, save_name):

    # exp_dir = 'exp/'
    # exp_list = os.listdir(exp_list)
    save_dir = 'sorted_result/'
    save_dir = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    # sort_target_list = glob.glob(os.path.join(exp_dir, '*', sort_target ))
    sort_target_list = glob.glob(os.path.join(exp_dir, sort_target ))
    exp_name_list = list()
    exp_dir = os.path.normcase(exp_dir)
    if os.path.basename(exp_dir) != '':
        exp_dir = exp_dir + os.path.sep
    for sort_target in sort_target_list:
        exp_name_list.append(sort_target.replace(exp_dir, '').split(os.path.sep)[0])
    print(sort_target_list)
    for exp_name, sort_target in zip(exp_name_list, sort_target_list):
        # exp_name = sort_target.split(os.path.sep)[0]
        sorted_dir = os.path.join(save_dir, exp_name +'_'+ os.path.basename(sort_target))
        shutil.copy(sort_target, sorted_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Sort specified "target" written in "Unix style pathname" in "sorted_result"')
    parser.add_argument('--target', type = str, help = 'filename to copy. have to specify relative file directory from each experiment directory. ex) validation/mcd.*')
    parser.add_argument('--exp_dir', default = 'exp/', type = str, help = 'Change base exp directory. Dafault to exp/')
    parser.add_argument('--save_dir', type = str)
    args = parser.parse_args()

    if args.save_dir == None:
        args.save_dir = time.strftime('%m%d_%H%M%S')
    sort_results(sort_target=args.target, exp_dir = args.exp_dir, save_name = args.save_dir)

# sort_target='mcd.*'
# s = sort_target_list[0]
# os.path.split(s)
# os.path.basename(s)
# os.path.dirname(s)
# os.path.splitext(s)
# os.path.splitdrive(s)
# help(os.path.altsep)
# os.path.defpath
# os.path.pathsep
# os.path.pardir
# s.split(os.path.sep)
# s
# help(os.path.splitdrive)
#
#
# for exp in exp_list:
#     file_dir = os.path.join(exp_dir, exp, sort_target)
#     shutil.copy()
#
# glob.glob('exp/*/mcd.*')
# glob.glob('exp/*/*')
# glob.glob('*.*')
# glob.glob('exp/*')
