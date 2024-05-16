import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of the results')
    parser.add_argument('--file_type', type=str, default='pdf', help='File type of the figures')

    args = parser.parse_args()

    file_type = args.file_type
    # ================================== EXP 1 ===================================
    # Convergence analysis
    ffo_result, cvxpylayer_result = {}, {}
    ffo_result_mean, cvxpylayer_result_mean = {}, {}
    ffo_result_std, cvxpylayer_result_std = {}, {}

    grad_differences = {}

    eps = 0.01
    ydim_list = [5, 10, 20, 50, 100, 200, 500] # , 800, 1000] # list(range(100,1000,100))
    directory_path = 'exp1_bilinear/'
    seed_list = list(set(range(1,31,1)) - set([2,9,29]))
    for ydim in ydim_list:
        directory_name = directory_path + 'ydim{}'.format(ydim)
        # Initialize the dictionary
        ffo_result[ydim], cvxpylayer_result[ydim] = [], []

        grad_differences[ydim] = []

        # Read the results
        ffo_filename_list                 = ['ffo_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
        ffo_grad_filename_list            = ['ffo_eps{}_seed{}.pickle'.format(eps, seed) for seed in seed_list]
        cvxpylayer_filename_list          = ['cvxpylayer_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
        cvxpylayer_gradient_filename_list = ['cvxpylayer_eps{}_seed{}.pickle'.format(eps, seed) for seed in seed_list]
        for filename in ffo_filename_list:
            df = pd.read_csv('results/' + directory_name + '/' + filename, header=None, names=['iteration', 'loss', 'time', 'time1', 'time2'], skiprows=1)
            ffo_result[ydim].append(df.values[:,:3])
        for filename in cvxpylayer_filename_list:
            df = pd.read_csv('results/' + directory_name + '/' + filename)
            cvxpylayer_result[ydim].append(df.values[:,:3])

        ffo_result_mean[ydim] = np.mean(ffo_result[ydim], axis=0)
        ffo_result_std[ydim] = np.std(ffo_result[ydim], axis=0)
        cvxpylayer_result_mean[ydim] = np.mean(cvxpylayer_result[ydim], axis=0)
        cvxpylayer_result_std[ydim] = np.std(cvxpylayer_result[ydim], axis=0)

        # Compute gradient difference        
        for filename in ffo_grad_filename_list:
            ffo_grad_list       = pickle.load(open('results/' + directory_name + '/' + filename, 'rb'))
        for filename in cvxpylayer_gradient_filename_list:
            cvxpylaer_grad_list = pickle.load(open('results/' + directory_name + '/' + filename, 'rb'))

        for ffo_grad, cvxpylayer_grad in zip(ffo_grad_list, cvxpylaer_grad_list):
            grad_differences[ydim].append(np.linalg.norm(ffo_grad - cvxpylayer_grad))

        # Plot the loss results
        data_type = 'loss'
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x_list = list(range(1, ffo_result_mean[ydim].shape[0] + 1))
        
        # Create a secondary y-axis
        sns.lineplot(x=x_list, y=ffo_result_mean[ydim][:,1], label='ffo', ax=ax1, linewidth=2.5, zorder=10)
        sns.lineplot(x=x_list, y=cvxpylayer_result_mean[ydim][:,1], label='cvxpylayer', ax=ax1, linewidth=2.5, zorder=5)
        ax1.set_ylabel('Loss', fontsize=28)
        ax1.legend(loc='upper right', fontsize=28, frameon=False)

        ax2 = ax1.twinx()
        sns.barplot(x=x_list, y=grad_differences[ydim], label='grad_diff', ax=ax2, zorder=3)
        ax2.set_ylabel('Gradient error', fontsize=28)

        ax1.set_xlabel('Iteration', fontsize=28)
        x_ticks = list(range(0, ffo_result_mean[ydim].shape[0], 50))
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks)

        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()
        ax2.set_ylim(bottom=0, top=0.5)

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        # plt.title('Convergence plot with y dimension = {}'.format(str(ydim)), fontsize=28)
        plt.xlabel('Iteration', fontsize=28)

        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles=handles[1:], labels=labels[1:], loc='upper right')

        # Save figure
        plt.tight_layout()
        plt.savefig('figures/' + directory_name + '_{}.{}'.format(data_type, file_type))
        plt.close()

    # ================================== EXP 2 ===================================
    # Computation cost analysis
    ffo_result, cvxpylayer_result = {}, {}
    ffo_result_mean, cvxpylayer_result_mean = {}, {}
    ffo_result_std, cvxpylayer_result_std = {}, {}

    eps = 0.01
    ydim_list = list(range(100,1100,100))
    directory_path = 'exp2_bilinear/'
    seed_list = list(set(range(1,11,1))) #- set([1, 10,11,12,13,14,18,19,20]))
    for ydim in ydim_list:
        directory_name = directory_path + 'ydim{}'.format(ydim)
        # Initialize the dictionary
        ffo_result[ydim], cvxpylayer_result[ydim] = [], []

        # Read the results
        cvxpylayer_filename_list = ['cvxpylayer_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
        ffo_filename_list        = ['ffo_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
        for filename in cvxpylayer_filename_list:
            try:
                df = pd.read_csv('results/' + directory_name + '/' + filename)
                cvxpylayer_result[ydim].append(df.values[:,:3])
            except:
                print(ydim, filename)
        for filename in ffo_filename_list:
            try:
                df = pd.read_csv('results/' + directory_name + '/' + filename, header=None, names=['iteration', 'loss', 'time', 'time1', 'time2'], skiprows=1)
                ffo_result[ydim].append(df.values[:,:3])
            except:
                print(ydim, filename)

        ffo_result_mean[ydim] = np.mean(ffo_result[ydim], axis=0)
        cvxpylayer_result_mean[ydim] = np.mean(cvxpylayer_result[ydim], axis=0)

    # Plot the time results
    time_results = {'ffo': [], 'cvxpylayer': [], 'ydim': []}
    for ydim in ffo_result.keys():
        time_results['ffo'].append(np.mean(ffo_result_mean[ydim][:,2]))
        time_results['cvxpylayer'].append(np.mean(cvxpylayer_result_mean[ydim][:,2]))
        time_results['ydim'].append(ydim)

    time_results = pd.DataFrame(time_results)
    plt.figure(figsize=(10, 6))
    g = sns.barplot(x='ydim', y='value', hue='variable', data=pd.melt(time_results, ['ydim']))
    g.set_xlabel('Inner level dimension', fontsize=28)
    g.set_ylabel('Time (s)', fontsize=28)

    # Set tick label size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust the legend
    handles, labels = g.get_legend_handles_labels()
    labels = [label.replace('variable', '') for label in labels]  # Remove "variable" from legend labels
    g.legend(handles=handles, labels=labels, title='', fontsize=28, frameon=False)
    
    plt.tight_layout()
    g.figure.savefig('figures/' + directory_path + 'time_results.{}'.format(file_type))
    plt.close()

    # ================================== EXP 3 ===================================
    # Epsilon analysis
    # Convergence analysis
    ffo_result, cvxpylayer_result = {}, {}
    ffo_result_mean, cvxpylayer_result_mean = {}, {}
    ffo_result_std, cvxpylayer_result_std = {}, {}
    grad_differences = {}

    eps = 0.01
    ydim_list = [100, 200, 500] # list(range(100,1000,100))
    directory_path = 'exp3_bilinear/'
    seed_list = list(range(1,11,1))
    for ydim in ydim_list:
        # Initialize the dictionary
        ffo_result[ydim], cvxpylayer_result[ydim] = {}, {}
        ffo_result_mean[ydim], cvxpylayer_result_mean[ydim] = {}, {}
        ffo_result_std[ydim], cvxpylayer_result_std[ydim] = {}, {}
        grad_differences[ydim] = {}

        # Plot the loss results
        data_type = 'gradient_error'
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Some random seed didn't finish
        # if ydim == 100:
        #     seed_list = [1,2,3,4,5]
        # elif ydim == 200:
        #     seed_list = [1,3,4,5]
        # elif ydim == 500:
        #     seed_list = [2,3,4,5]
        # else:
        #     seed_list = [1,2,3,4,5]

        for eps in [0.0001, 0.001, 0.01, 0.1, 1.0]:
            ffo_result[ydim][eps], cvxpylayer_result[ydim][eps] = [], []
            grad_differences[ydim][eps] = []

            directory_name = directory_path + 'ydim{}'.format(ydim)

            # Read the results
            ffo_filename_list                 = ['ffo_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
            ffo_grad_filename_list            = ['ffo_eps{}_seed{}.pickle'.format(eps, seed) for seed in seed_list]
            cvxpylayer_filename_list          = ['cvxpylayer_eps{}_seed{}.txt'.format(eps, seed) for seed in seed_list]
            cvxpylayer_gradient_filename_list = ['cvxpylayer_eps{}_seed{}.pickle'.format(eps, seed) for seed in seed_list]
            for filename in ffo_filename_list:
                df = pd.read_csv('results/' + directory_name + '/' + filename, header=None, names=['iteration', 'loss', 'time', 'time1', 'time2'], skiprows=1)
                ffo_result[ydim][eps].append(df.values[:,:3])
            # for filename in cvxpylayer_filename_list:
            #     df = pd.read_csv('results/' + directory_name + '/' + filename)
            #     cvxpylayer_result[ydim][eps].append(df.values[:,:3])

            ffo_result_mean[ydim][eps] = np.mean(ffo_result[ydim][eps], axis=0)
            cvxpylayer_result_mean[ydim][eps] = np.mean(cvxpylayer_result[ydim][eps], axis=0)

            x_list = list(range(1, ffo_result_mean[ydim][eps].shape[0] + 1))
            # Create a secondary y-axis
            sns.lineplot(x=x_list, y=ffo_result_mean[ydim][eps][:,1], label='ffo ' + r'$\alpha^2$' + '={}'.format(eps), ax=ax1, linewidth=2.5)

        ax1.set_ylabel('Loss', fontsize=28)
        ax1.legend(loc='upper right', fontsize=28, frameon=False)

        ax1.set_xlabel('Iteration', fontsize=28)
        x_ticks = list(range(0, ffo_result_mean[ydim][eps].shape[0], 50))
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks)

        # plt.title('Convergence of different gradient accuracy', fontsize=28)
        plt.xlabel('Iteration', fontsize=28)

        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles=handles[1:], labels=labels[1:], loc='upper right')

        # Save figure
        plt.tight_layout()
        plt.savefig('figures/' + directory_name + '_{}.{}'.format(data_type, file_type))
        plt.close()

