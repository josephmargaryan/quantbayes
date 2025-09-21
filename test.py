import numpy as np
from scipy.optimize import linprog

k = 3
eps = np.log(9.0)  # epsilon = ln 9
E = np.exp(eps)  # == 9.0

# Variables: x = [p, q]. We maximize p -> minimize -p
c = np.array([-1.0, 0.0])

# Inequalities (A_ub x <= b_ub):
# 1) p - E q <= 0   (i.e., p <= E q)  -- binding at optimum
# 2) q - E p <= 0   (i.e., q <= E p)  -- redundant here but fine to include
A_ub = np.array([[1.0, -E], [-E, 1.0]])
b_ub = np.array([0.0, 0.0])

# Equality: p + (k-1) q = 1
A_eq = np.array([[1.0, k - 1.0]])
b_eq = np.array([1.0])

# Bounds: p >= 0, q >= 0
bounds = [(0.0, None), (0.0, None)]

res = linprog(
    c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
)
print("status:", res.message)
p, q = res.x
print("p =", p, "q =", q)  # expect p=9/11, q=1/11


import math


def krr_params(k: int, eps: float):
    E = math.exp(eps)
    p = E / (E + k - 1.0)
    q = 1.0 / (E + k - 1.0)
    return p, q  # Pr(correct), Pr(each wrong class)


# Example: k=3, eps=ln 9
p, q = krr_params(3, math.log(9.0))  # -> p=9/11, q=1/11

import random

# fixed encoding from classes to bitstrings with no '11'
ENC = {"A": "00", "B": "01", "C": "10"}
CLASSES = ["A", "B", "C"]


def privatize_class(x: str, eps: float):
    k = len(CLASSES)
    p, q = krr_params(k, eps)
    probs = [q] * k
    probs[CLASSES.index(x)] = p
    y = random.choices(CLASSES, weights=probs, k=1)[0]
    return ENC[y]  # deterministic post-processing
