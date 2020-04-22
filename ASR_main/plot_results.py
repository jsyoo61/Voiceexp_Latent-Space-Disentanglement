import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import glob
from tools.tools import load_pickle, save_pickle, read

def sort_results(sort_target, exp_dir, save_name):
    save_dir = 'sorted_result/'
    save_dir = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
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



def plot_result(exp_name):
    exp_dir = 'exp'
    exp_dir = os.path.join(exp_dir, exp_name)
    validation_dir = 'validation/'
    validation_dir = os.path.join(exp_dir, validation_dir)
    validation_result_dir = 'validation_result/'
    validation_result_dir = os.path.join(validation_dir, validation_result_dir)
    validation_result_list = os.listdir(validation_result_dir)

    mean = list()
    # std = list()
    epoch = list()
    for validation_result in validation_result_list:
        epoch.append(int(validation_result.split('.')[0].split('_')[-1])) # Filename: validation_1.p
        validation_result_1 = os.path.join(validation_result_dir, validation_result)
        validation_result = load_pickle(validation_result_1)
        mean.append(validation_result.mean().values)
        # std.append(validation_result.std().values)

    mean = pd.DataFrame(mean, columns = ['mcd','msd','gv'], index = epoch).sort_index()
    save_pickle(mean, 'mean.p')

    for measure in mean.columns:
        fig_save_dir = os.path.join(validation_dir, measure+'.png')
        axes = mean.plot(y=measure, style='o-')
        fig = axes.get_figure()
        fig.savefig(fig_save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot&Save validation results with specified exp_name')
    parser.add_argument('--exp_name', type = str, help = 'Experiment name. All files will be stored in exp/exp_name')
    args = parser.parse_args()

    plot_result(exp_name = args.exp_name)

# p=load_pickle('mean.p')
# p.sort_index()
# p
# p=p.sort_index()
# ax = p.plot(y='gv', style='o-')
# p=load_pickle('validation_1.p')
# for i in p.index:
#     print(i)
# a


# c
#
# for i in c:
#     print(i)
#
# fig = ax.get_figure()
# fig.savefig('test.jpg')
# fig = plt.figure()
# fig.add_axes(ax)
# ax = c.plot()
# ax.savefig
# help(c.plot)
# c.plot(y='e')
# import numpy as np
# c.loc['23'] = np.arange(6)
# c.loc[2] = np.arange(4,10)
# a = validation_result.mean()
# b = validation_result.std()
# c = pd.DataFrame(columns = ['a','b','c','d','e','f'])
# c.append(pd.Series(pd.concat([a,b]), name = '123'))
# pd.Series(pd.concat([a,b]), name = '123').reindex(['a','b','c','d','e','f'])
# pd.concat([a,b]).reindex(['a','b','c','d','e','f'])
# a.append(b, )
# a.to_frame().rename(columns=['a','b','c'])
# a.values
# a.plot()
# c.append(a.to_frame().T)
#
# a.to_frame.
# help(a.append)
# help(pd.Series)
# b
#     # mean.append(validation_result.mean().to_frame().T)
#     # std.append(validation_result.std().to_frame().T)
# pd.DataFrame(validation_result.mean().to_frame().T, index = [0,1])
#
# help(pd.Series.to_frame)
# help(pd.concat)
# mean = pd.concat(mean, axis = 1, keys=epoch).T
# mean
# epoch
# std = pd.concat(std, axis = 1)
# validation_result.mean()
# pd.concat(mean).index=epoch
# pd.concat(mean, keys = epoch)
# asd
# mean = pd.Concat(mean)
# std
#
# # Plot and save image
# for performance_measure in mean.columns:
#     plt.plot(epoch, mean[performance_measure])
#
#     save plot
