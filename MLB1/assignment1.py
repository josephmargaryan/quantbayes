import numpy as np
import matplotlib.pyplot as plt


# Hoeffding's inequality
def hoeffding(n=1000, delta=0.01):
    return np.sqrt((np.log(1 / delta)) / (2 * n))


# Binary kl-divergence
def kl_div(p_hat, q):
    if p_hat == 0:
        first_term = 0.0
    else:
        first_term = p_hat * np.log(p_hat / q)
    if p_hat == 1:
        second_term = 0.0
    else:
        second_term = (1 - p_hat) * np.log((1 - p_hat) / (1 - q))
    return first_term + second_term


# kl epsilon
def kl_epsilon(n=1000, delta=0.01):
    return np.log(1 / delta) / n


# Unique solution via binary search
def kl_upper(p_hat, n=1000, delta=0.01, tol=1e-10):
    if p_hat == 1.0:
        return 1.0
    epsilon = kl_epsilon(n, delta)
    lower, upper = p_hat, 1
    while (upper - lower) > tol:
        mid = (upper + lower) / 2
        div = kl_div(p_hat, mid)
        if div > epsilon:
            upper = mid
        else:
            lower = mid
    return lower


def pinsker(p_hat, n=1000, delta=0.01):
    return p_hat + np.sqrt((np.log(1 / delta)) / (2 * n))


def refined_pinskers(p_hat, n=1000, delta=0.01):
    first_term = p_hat + np.sqrt((2 * p_hat * np.log(1 / delta)) / (n))
    second_term = 2 * ((np.log(1 / delta)) / (n))
    return first_term + second_term


p_hats = np.linspace(0, 1, 1000)
hoeffding_bounds = p_hats + hoeffding()
kl_bounds = [kl_upper(p) for p in p_hats]
pinsker_bounds = [pinsker(p) for p in p_hats]
refined_pinsker_bounds = [refined_pinskers(p) for p in p_hats]

hoeffding_bounds = np.clip(hoeffding_bounds, 0, 1)
kl_bounds = np.clip(kl_bounds, 0, 1)
pinsker_bounds = np.clip(pinsker_bounds, 0, 1)
refined_pinsker_bounds = np.clip(refined_pinsker_bounds, 0, 1)


plt.plot(p_hats, hoeffding_bounds, label=r"Hoeffding")
plt.plot(p_hats, kl_bounds, label=r"kl")
plt.plot(p_hats, pinsker_bounds, label=r"Pinsker")
plt.plot(p_hats, refined_pinsker_bounds, label=r"Refined Pinsker")
plt.legend()
plt.ylabel(r"Upper confidence bound")
plt.xlabel(r"$\hat{p}_{n}$")
plt.title(
    r"All four bounds of a function of $\hat{p}_{n}$ for $\hat{p}_{n} \in [0, 1]$"
)
plt.show()

# Zoomed in plot
plt.plot(p_hats, hoeffding_bounds, label=r"Hoeffding")
plt.plot(p_hats, kl_bounds, label=r"kl")
plt.plot(p_hats, pinsker_bounds, label=r"Pinsker")
plt.plot(p_hats, refined_pinsker_bounds, label=r"Refined Pinsker")
plt.legend()
plt.ylabel(r"Upper confidence bound")
plt.xlabel(r"$\hat{p}_{n}$")
plt.title(r"Zoom: $\hat p_n\in[0,0.1]$ and bounds $\in [0, 0.2]$")
plt.xlim(0.0, 0.1)
plt.ylim(0.0, 0.2)
plt.show()


# Unique solution via binary search
def kl_lower(p_hat, n=1000, delta=0.01, tol=1e-10):
    if p_hat == 0.0:
        return 0.0
    epsilon = kl_epsilon(n, delta)
    lower, upper = 0, p_hat
    while (upper - lower) > tol:
        mid = (upper + lower) / 2
        div = kl_div(p_hat, mid)
        if div > epsilon:
            lower = mid
        else:
            upper = mid
    return upper


hoeffding_lower_bounds = p_hats - hoeffding()
kl_lower_bounds = [kl_lower(p) for p in p_hats]
hoeffding_lower_bounds = np.clip(hoeffding_lower_bounds, 0, 1)
kl_lower_bounds = np.clip(kl_lower_bounds, 0, 1)

plt.plot(p_hats, hoeffding_lower_bounds, label="Hoeffdings lower bounds")
plt.plot(p_hats, kl_lower_bounds, label="KL lower bounds")
plt.title(r"Hoeffding lower bound vs kl lower bound")
plt.legend()
plt.show()
