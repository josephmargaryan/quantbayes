# Decentralized Ball privacy

This README is the focused guide for the decentralized path.

## 1. Observer-specific privacy notion

Let node $j$ be the attacked node and $A$ the observer set.
The observer sees
```math
\text{View}_A(M(D)).
```
The decentralized privacy notion is observer-specific Ball-PN-RDP:
```math
D_\alpha\Big(
\text{View}_A(M(D))
\,\|\|\,
\text{View}_A(M(D'))
\Big)
\le
\varepsilon_{A\leftarrow j}(\alpha;r)
\qquad
\text{for } D\sim_{r,j} D'.
```
This captures how much information about a Ball-local change at node $j$ reaches observer set $A$.

---

## 2. Linear-Gaussian observer theorem

When the observer view can be written as
```math
Y_A(D)
=
c_A(D_{-j}) + (H_{A\leftarrow j}\otimes I)s_j(D)+\zeta_A,
\qquad
\zeta_A\sim\mathcal N(0,\Sigma_A),
```
with blockwise Ball sensitivities $\Delta_{j,\ell}(r)$, the theorem gives
```math
D_\alpha\big(Y_A(D)\|Y_A(D')\big)
\le
\frac\alpha2\,\Delta_{A\leftarrow j}(r)^2,
```
where
```math
\Delta_{A\leftarrow j}(r)^2
:=
\sup_{\|\delta_\ell\|\le \Delta_{j,\ell}(r)}
\Big\|\Sigma_A^{-1/2}(H_{A\leftarrow j}\otimes I)\delta\Big\|_2^2.
```
So the topology and observer view enter explicitly through $H_{A\leftarrow j}$.

---

## 3. Observer-specific Ball-ReRo

The generic Ball-PN-RDP $\Rightarrow$ Ball-ReRo theorem gives
```math
p_{\mathrm{succ}}(\eta)
\le
\min\left\{1,\,(\kappa(\eta)e^{\varepsilon_{A\leftarrow j}(\alpha;r)})^{(\alpha-1)/\alpha}\right\}.
```
For a uniform finite prior on $m$ candidates, $\kappa=1/m$.

So the decentralized finite-prior exact-identification result is still a scalar success-probability upper bound, but now it depends on the observer set through the observer-specific privacy curve.

---

## 4. Observer-specific exact MAP attack on a Gaussian view

Under the linear-Gaussian observer theorem, a MAP estimator solves
```math
\widehat z_A
\in
\arg\min_{z\in \mathcal B(u,r)}
\left\{
\frac12
\left\|
\Sigma_A^{-1/2}
\big(y_A-c_A(D^-)-(H_{A\leftarrow j}\otimes I)s_j(z)\big)
\right\|_2^2
-
\log \pi_{u,r}(z)
\right\}.
```
For a finite prior, this reduces to exact candidate scoring.

---

## 5. Practical use

The decentralized code is most useful when you already have:
- a node-local Ball sensitivity model;
- an explicit observer set;
- and either a public transcript or a linearized Gaussian observer view.

For the main centralized thesis notebook, you can ignore this path. For topology-aware experiments, start here.
