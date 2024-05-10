import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    ffo_result = {}
    cvxpylayer_result = {}

    # ================================== EXP 1 ===================================
    eps = 0.01
    ydim_list = list(range(100,1000,100))
    directory_path = 'exp1/'
    seed_list = [2,3,4,5]
    for ydim in ydim_list:
        directory_name = 'exp1/' + 'ydim{}'.format(ydim)
        # Initialize the dictionary
        ffo_result[ydim] = []
        cvxpylayer_result[ydim] = []

        # Read the results
        cvxpylayer_filename_list = ['cvxpylayer_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
        ffo_filename_list        = ['ffo_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
        for filename in cvxpylayer_filename_list:
            df = pd.read_csv('results/' + directory_name + '/' + filename)
            cvxpylayer_result[ydim].append(df.values[:,:3])
        for filename in ffo_filename_list:
            df = pd.read_csv('results/' + directory_name + '/' + filename, header=None, names=['iteration', 'loss', 'time', 'time1', 'time2'], skiprows=1)
            ffo_result[ydim].append(df.values[:,:3])

        ffo_result[ydim] = np.mean(ffo_result[ydim], axis=0)
        cvxpylayer_result[ydim] = np.mean(cvxpylayer_result[ydim], axis=0)

        # Plot the loss results
        data_type = 'loss'
        plt.figure()
        plt.plot(ffo_result[ydim][:,1], label='ffo')
        plt.plot(cvxpylayer_result[ydim][:,1], label='cvxpylayer')
        plt.legend()
        plt.title('ydim = ' + str(ydim))
        plt.savefig('figures/' + directory_name + '_{}.png'.format(data_type))
        plt.close()

    # Plot the time results
    time_results = {'ffo': [], 'cvxpylayer': [], 'ydim': []}
    for ydim in ffo_result.keys():
        time_results['ffo'].append(np.mean(ffo_result[ydim][:,2]))
        time_results['cvxpylayer'].append(np.mean(cvxpylayer_result[ydim][:,2]))
        time_results['ydim'].append(ydim)

    time_results = pd.DataFrame(time_results)
    g = sns.barplot(x='ydim', y='value', hue='variable', data=pd.melt(time_results, ['ydim']))
    g.set_title('Time results')
    g.figure.savefig('figures/' + directory_path + 'time_results.png')

    # ================================== EXP 2 ===================================



    # ================================== EXP 3 ===================================



