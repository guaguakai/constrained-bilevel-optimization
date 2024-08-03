## Fully First-Order Constrained Bilevel Optimization


This repo includes the implementation of the fully first-order gradient descent algorithm for constrained bilevel optimization.

### Dependencies
- CvxpyLayer: https://github.com/cvxgrp/cvxpylayers
- Pytorch
- Numpy

### Linear inequality constraints
- Non-fully first order method (implemented using CvxpyLayer):
` python3 main.py --solver=cvxpylayer --seed=0 --ydim=20 --n-constraints=5`

- Fully first-order method:
`python3 main.py --solver=ffo --seed=0 --ydim=20 --n-constraints=5`

### Bilinear inequality constraints
The same analysis and approach work for the bilinear case. The bilinear case is implemented in `main_bilinear.py`. You can also run the non-fully first order and fully first-order methods for the bilinear case using the following commands:

- Non-fully first order method (implemented using CvxpyLayer):
` python3 main_bilinear.py --solver=cvxpylayer --seed=0 --ydim=20 --n-constraints=5`

- Fully first-order method:
`python3 main_bilinear.py --solver=ffo --seed=0 --ydim=20 --n-constraints=5`


You can change the optimizer inside `main.py`
