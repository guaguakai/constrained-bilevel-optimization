## Fully First-Order Constrained Bilevel Optimization


This repo includes the implementation of the fully first-order gradient descent algorithm for constrained bilevel optimization.


Second-order method:
` python3 main.py --solver=cvxpylayer --seed=0 --ydim=20 --n-constraints=5`

Fully first-order method:
`python3 main.py --solver=ffo --seed=0 --ydim=20 --n-constraints=5`

You can change the optimizer inside `main.py`
