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
    c = c / torch.norm(c) # normalize it
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
    
    # Define functions (pytorch versions)
    f = lambda x,y: c @ y + 0.01 * torch.norm(x)**2 + 0.01 * torch.norm(y)**2
    g = lambda x,y: 1/2 * y.t() @ Q @ y + x.t() @ P @ y
    h = lambda x,y: A @ y - b 

    # Compute gradient using pytorch
    x = torch.rand(x_dim, requires_grad=True) # random initialization
    optimizer = torch.optim.Adam([x], lr=lr)
    # optimizer = torch.optim.SGD([x], lr=lr)

    # Output file
    f_output = open('results/ydim{}/{}_seed{}.txt'.format(y_dim, solver, seed), 'w')
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
            optimizer.zero_grad()
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
    
            loss = f(x, torch.tensor(y_cp.value).float())  # + 1/2 * x @ x
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            time_elapsed = time.time() - start_time
            print(i, loss.item(), time_elapsed)
            f_output.write('{}, {}, {} \n'.format(i, loss, time_elapsed))

    elif solver == 'ffo':
        # Our algorithm
        epsilon = 0.01
        lamb_init = 1 / epsilon
        lamb = lamb_init
        delta = torch.zeros(x_dim)
        for i in range(args.n_iterations):
            start_time = time.time()
            # Perturbed x to get xx
            # s = torch.rand(x_dim)
            s = torch.normal(torch.zeros(x_dim), torch.ones(x_dim)) * 0.01
            xx = x + s

            # Pre-solve inner optimization problem:  min_y max_\gamma g(x,y) + \gamma^\top h(x,y)
            x_cp = xx.detach().numpy()
            y_cp = cp.Variable(y_dim)
            q_cp = x_cp.T @ P_cp
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_cp) + q_cp @ y_cp)
            constraints = [A_cp @ y_cp - b_cp <= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(eps_abs=1e-12, max_iter=100000)

            y_opt = y_cp.value
            gamma_opt = constraints[0].dual_value # We only have one set of constraints

            # print('optimal y:', y_opt)
            # print('optimal gamma:', gamma_opt)
    
            # Solve Lagrangian optimization problem: min_y f(x,y) + \lambda ( g(x,y) + \gamma^\top h(x,y) - g(x,y*) - \gamma^\top h(x,y*) ) + 1/2 * \lambda^2 * |h(x,y*)|^2
            y_cp = cp.Variable(y_dim)
            f_cp = c_cp.T @ y_cp + 0.01 * cp.sum_squares(x_cp) + 0.01 * cp.sum_squares(y_cp)
            g_cp = 0.5 * cp.quad_form(y_cp, Q_cp) + q_cp @ y_cp
            g_opt_cp = 0.5 * cp.quad_form(y_opt, Q_cp) + q_cp @ y_opt
            h_cp = A_cp @ y_cp - b_cp
            h_opt_cp = A_cp @ y_opt - b_cp
            objective = cp.Minimize( f_cp + lamb * (g_cp + gamma_opt.T @ h_cp - g_opt_cp - gamma_opt.T @ h_opt_cp) + 0.5 * lamb**2 * cp.sum_squares(cp.maximum(h_cp, 0))  )
            problem = cp.Problem(objective, constraints)
            problem.solve(eps_abs=1e-12, max_iter=100000)
            
            y_lamb_opt = y_cp.value

            # Convert numpy to tensor
            y_opt = torch.tensor(y_opt).float()
            y_lamb_opt = torch.tensor(y_lamb_opt).float()
            gamma_opt = torch.tensor(gamma_opt).float()

            # Compute the final Lagrangian and track gradient over x
            final_lagrangian = f(xx,y_lamb_opt) + lamb * (g(xx, y_lamb_opt) + gamma_opt.T @ h(xx, y_lamb_opt) - g(xx, y_opt) - gamma_opt.T @ h(xx, y_opt)) + 0.5 * lamb**2 * torch.sum(torch.clip(h(xx, y_lamb_opt), min=0) **2)
            # final_lagrangian.backward()
            # x.grad = torch.clamp(x.grad, max=1, min=-1)

            # Gradient and variable update
            D = 1
            gradient = torch.autograd.grad(final_lagrangian, xx)[0]
            with torch.no_grad():
                # x += delta
                # x -= gradient * lr
                # x.grad = -delta
                x.grad = gradient
            # delta = gradient * lr # torch.clamp(delta - lr * gradient, min=-D, max=D)
            optimizer.step()
            optimizer.zero_grad()

            # Compute hyperobjective (this is not accurate but should be close enough)
            loss = f(xx, y_opt).item()

            # lambda update # TODO
            lamb = lamb_init

            # Output file
            time_elapsed = time.time() - start_time
            print(i, loss, time_elapsed)
            f_output.write('{}, {}, {} \n'.format(i, loss, time_elapsed))

    f_output.close()
