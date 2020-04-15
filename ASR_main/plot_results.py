import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tools.tools import load_pickle, save_pickle, read

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
# p=load_pickle('mean.p')
# p.sort_index()
# p
# p=p.sort_index()
# ax = p.plot(y='gv', style='o-')
# p=load_pickle('validation_1.p')
# for i in p.index:
#     print(i)
# a

    for measure in mean.columns:
        fig_save_dir = os.path.join(validation_dir, measure+'.png')
        axes = mean.plot(y=measure, style='o-')
        fig = axes.get_figure()
        fig.savefig(fig_save_dir)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot&Save validation results with specified exp_name')
    parser.add_argument('--exp_name', type = str, help = 'Experiment name. All files will be stored in exp/exp_name')
    args = parser.parse_args()

    plot_result(exp_name = args.exp_name)
