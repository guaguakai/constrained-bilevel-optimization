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
    parser.add_argument('--solver', default = 'ffo', choices=['ffo', 'cvxpylayer'])
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--n-iterations', type=int, default=1000)

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    lr = args.lr
    solver = args.solver

    # Parameters
    x_dim = 200
    y_dim = 200
    n_constraints = 100

    c = torch.rand(x_dim)
    L = torch.rand(y_dim, y_dim)
    Q = L.t() @ L # PSD matrix
    P = torch.rand(x_dim, y_dim)
    A = torch.rand(n_constraints, y_dim) # A y - b \leq 0
    b = torch.rand(n_constraints)

    c_cp = c.numpy()
    Q_cp = Q.numpy()
    P_cp = P.numpy()
    A_cp = A.numpy()
    b_cp = b.numpy()
    
    # Define functions
    f = lambda x,y: c @ y 
    g = lambda x,y: 1/2 * y.t() @ Q @ y + x.t() @ P @ y
    h = lambda x,y: A @ y - b 

    # Compute gradient using pytorch
    x = torch.rand(x_dim, requires_grad=True) # random initialization
    optimizer = torch.optim.Adam([x], lr=lr)

    if solver == 'cvxpylayer':
        # Cvxpy (differentiable optimization) approach
        for i in range(args.n_iterations):
            y_cp = cp.Variable(y_dim)
            x_cp = cp.Parameter(x_dim)
            objective = cp.Minimize(0.5 * cp.quad_form(y_cp, Q_cp) + x_cp.T @ P_cp @ y_cp)
            constraints = [A @ y_cp - b <= 0]
            problem = cp.Problem(objective, constraints)
            cvxpylayer = CvxpyLayer(problem, parameters=[x_cp], variables=[y_cp])
    
            solution, = cvxpylayer(x)
            loss = f(x, solution)  # + 1/2 * x @ x
            loss.backward()
            optimizer.step()
            print(i, loss.item())

    elif solver == 'ffo':
        # Our algorithm
        lamb_init = 0.01
        for i in range(args.n_iterations):
            # Pre-solve inner optimization problem:  min_y max_\gamma g(x,y) + \gamma^\top h(x,y)
            x_detach = x.detach()
            y_init = torch.rand(y_dim)
            gamma_init = torch.clip(torch.rand(n_constraints), min=0)
            y_opt = torch.tensor(y_init, requires_grad=True)
            gamma_opt = torch.tensor(gamma_init, requires_grad=True)
    
            for j in range(args.steps):
                obj_inner = g(x_detach, y_opt) + gamma_opt.t() @ h(x_detach, y_opt)
                obj_inner.backward()
                y_opt = y_opt - lr * y_opt.grad # How to satisfy constraints? Projection? I don't want to implement convex projection though  TODO
                gamma_opt = torch.clip(gamma_opt + lr * gamma_opt.grad, min=0)
    
            # lambda update # TODO
            lamb = lamb_init
    
            # Define Lagrangian
            # L = f(x,y_lamb) + lamb * (g(x, y_lamb + gamma_opt.t() @ h(x, y_lamb) - g(x, y_opt) - gamma_opt.t() @ h(x, y_opt)) + 1/2 * lamb**2 * torch.sum(torch.clip(h(x,y_lamb), min=0) ** 2)
    
            # Minimize Lagrangian: solve y_lamb = \argmin L(x,y_lamb)
            y_lamb = torch.tensor(y_init, requires_grad=True)
            for j in range(args.steps):
                lagrangian = f(x_detach, y_lamb) + lamb * (g(x_detach, y_lamb) + gamma_opt.t() @ h(x_detach, y_lamb) - g(x_detach, y_opt) - gamma_opt.t() @ h(x_detach, y_opt)) + 1/2 * lamb**2 * torch.sum(torch.clip(h(x_detach,y_lamb), min=0) ** 2)
                lagrangian.backward()
                y_lamb = y_lamb - lr * y_lamb.grad # Projection # TODO

            # Compute the final Lagrangian and track gradient over x    
            final_lagrangian = f(x,y_lamb) + lamb * (g(x, y_lamb) + gamma_opt.t() @ h(x, y_lamb) - g(x, y_opt) - gamma_opt.t() @ h(x, y_opt)) + 1/2 * lamb**2 * torch.sum(torch.clip(h(x,y_lamb), min=0) ** 2)
            final_lagrangian.backward()
            optimizer.step()
            loss = f(x, y_lamb).item()
            print(i, loss) 
