import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    BS_result = {}
    ffo_result = {}
    cvxpylayer_result = {}

    for directory_name in os.listdir('results'):
        ydim = int(directory_name[4:])
        if ydim >= 500:
            continue
        else:
            # Initialize the dictionary
            BS_result[ydim] = []
            ffo_result[ydim] = []
            cvxpylayer_result[ydim] = []

            # Read the results
            for filename in os.listdir('results/' + directory_name):
                solver, eps, seed = filename.split('_')
                eps = eps[3:]
                seed = int(seed[4:])
                df = pd.read_csv('results/' + directory_name + '/' + filename)
                if solver == 'BR':
                    BS_result[ydim].append(df.values)
                elif solver == 'ffo':
                    ffo_result[ydim].append(df.values)
                elif solver == 'cvxpylayer':
                    cvxpylayer_result[ydim].append(df.values)
                else:
                    pass
                    # raise ValueError('Invalid filename')

            BS_result[ydim] = np.mean(BS_result[ydim], axis=0)
            ffo_result[ydim] = np.mean(ffo_result[ydim], axis=0)
            cvxpylayer_result[ydim] = np.mean(cvxpylayer_result[ydim], axis=0)

        # Plot the loss results
        data_type = 'loss'
        plt.figure()
        # plt.plot(BS_result[ydim][:,1], label='BS')
        plt.plot(ffo_result[ydim][:,1], label='ffo')
        plt.plot(cvxpylayer_result[ydim][:,1], label='cvxpylayer')
        plt.legend()
        plt.title('ydim = ' + str(ydim))
        plt.savefig('figures/' + directory_name + '_{}.png'.format(data_type))
        plt.close()

    # Plot the time results
    time_results = {'BS': [], 'ffo': [], 'cvxpylayer': [], 'ydim': []}
    for ydim in BS_result.keys():
        time_results['BS'].append(np.mean(BS_result[ydim][:,2]))
        time_results['ffo'].append(np.mean(ffo_result[ydim][:,2]))
        time_results['cvxpylayer'].append(np.mean(cvxpylayer_result[ydim][:,2]))
        time_results['ydim'].append(ydim)

    time_results = pd.DataFrame(time_results)
    g = sns.barplot(x='ydim', y='value', hue='variable', data=pd.melt(time_results, ['ydim']))
    g.set_title('Time results')
    g.figure.savefig('figures/time_results.png')
