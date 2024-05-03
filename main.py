import torch
import argparse
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
# import qpth
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default = 'ffo', choices=['ffo', 'BR', 'cvxpylayer'])
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--xdim', default = 100, type = int)
    parser.add_argument('--ydim', default = 500, type = int)
    parser.add_argument('--n-constraints', default = 100, type = int)
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--n-iterations', type=int, default=1000)

    args = parser.parse_args()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    lr = args.lr
    solver = args.solver

    # Parameters
    x_dim = args.xdim
    y_dim = args.ydim
    n_constraints = args.n_constraints

    c = torch.rand(y_dim)
    L = torch.rand(y_dim, y_dim)
    reg = 1
    Q = L.t() @ L + reg * torch.eye(y_dim) # PSD matrix
    P = torch.rand(x_dim, y_dim)
    A = torch.rand(n_constraints, y_dim) # A y - b \leq 0
    b = torch.rand(n_constraints)

    c_cp = c.numpy()
    Q_cp = Q.numpy()
    P_cp = P.numpy()
    A_cp = A.numpy()
    b_cp = b.numpy()
    
    # Define functions
    f = lambda x,y: c @ y + 0.01 * torch.norm(x) + 0.01 * torch.norm(y)
    g = lambda x,y: 1/2 * y.t() @ Q @ y + x.t() @ P @ y
    h = lambda x,y: A @ y - b 

    # Compute gradient using pytorch
    x = torch.rand(x_dim, requires_grad=True) # random initialization
    optimizer = torch.optim.Adam([x], lr=lr)

    # Output file
    f_output = open('results/ydim{}/{}_seed{}.txt'.format(ydim, solver, seed), 'w')
    f_output.write('Iteration, loss, time \n')

    if solver == 'cvxpylayer':
        # Cvxpy (differentiable optimization) approach
        for i in range(args.n_iterations):
            start_time = time.time()
            x_cp = cp.Parameter(x_dim)
            y_cp = cp.Variable(y_dim)
            q_cp = x_cp.T @ P_cp
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_cp) + q_cp @ y_cp)
            constraints = [A @ y_cp - b <= 0]
            problem = cp.Problem(objective, constraints)
            cvxpylayer = CvxpyLayer(problem, parameters=[x_cp], variables=[y_cp])
    
            solution, = cvxpylayer(x)
            loss = f(x, solution)  # + 1/2 * x @ x
            loss.backward()
            optimizer.step()
            time_elapsed = time.time() - start_time
            print(i, loss.item(), time_elapsed)
            f_output.write('{}, {}, {} \n'.format(i, loss, time_elapsed))

    elif solver == 'BR':
        for i in range(args.n_iterations):
            start_time = time.time()
            x_cp = x.detach().numpy()
            y_cp = cp.Variable(y_dim)
            q_cp = x_cp.T @ P_cp
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_cp) + q_cp @ y_cp)
            constraints = [A @ y_cp - b <= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve()
    
            loss = f(x, y_cp.value)  # + 1/2 * x @ x
            loss.backward()
            optimizer.step()

            time_elapsed = time.time() - start_time
            print(i, loss.item(), time_elapsed)
            f_output.write('{}, {}, {} \n'.format(i, loss, time_elapsed))

    elif solver == 'ffo':
        # Our algorithm
        lamb_init = 1
        lamb = lamb_init
        for i in range(args.n_iterations):
            start_time = time.time()
            # Pre-solve inner optimization problem:  min_y max_\gamma g(x,y) + \gamma^\top h(x,y)
            x_cp = x.detach().numpy()
            y_cp = cp.Variable(y_dim)
            q_cp = x_cp.T @ P_cp
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_cp) + q_cp @ y_cp)
            constraints = [A_cp @ y_cp - b_cp <= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve()

            y_opt = y_cp.value
            gamma_opt = constraints[0].dual_value # We only have one set of constraints

            # print('optimal y:', y_opt)
            # print('optimal gamma:', gamma_opt)
    
            # Solve Lagrangian optimization problem: min_y f(x,y) + \lambda ( g(x,y) + \gamma^\top h(x,y) - g(x,y*) - \gamma^\top h(x,y*) ) + 1/2 * \lambda^2 * |h(x,y*)|^2
            y_cp = cp.Variable(y_dim)
            f_cp = c_cp.T @ y_cp
            g_cp = 0.5 * cp.quad_form(y_cp, Q_cp) + q_cp @ y_cp
            g_opt_cp = 0.5 * cp.quad_form(y_opt, Q_cp) + q_cp @ y_opt
            h_cp = A_cp @ y_cp - b_cp
            h_opt_cp = A_cp @ y_opt - b_cp
            objective = cp.Minimize( f_cp + lamb * (g_cp + gamma_opt.T @ h_cp - g_opt_cp - gamma_opt.T @ h_opt_cp) + 0.5 * lamb**2 * cp.sum_squares(cp.maximum(h_cp, 0))  )
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            y_lamb_opt = y_cp.value

            # Convert numpy to tensor
            y_opt = torch.tensor(y_opt).float()
            y_lamb_opt = torch.tensor(y_lamb_opt).float()
            gamma_opt = torch.tensor(gamma_opt).float()

            # Compute the final Lagrangian and track gradient over x
            final_lagrangian = f(x,y_lamb_opt) + lamb * (g(x, y_lamb_opt) + gamma_opt.T @ h(x, y_lamb_opt) - g(x, y_opt) - gamma_opt.T @ h(x, y_opt)) + 0.5 * lamb**2 * torch.sum(torch.clip(h(x, y_lamb_opt), min=0) **2)
            final_lagrangian.backward()
            optimizer.step()
            loss = f(x, y_opt).item()

            # lambda update # TODO
            lamb = lamb_init

            # Output file
            time_elapsed = time.time() - start_time
            print(i, loss, time_elapsed)
            f_output.write('{}, {}, {} \n'.format(i, loss, time_elapsed))

    f_output.close()
