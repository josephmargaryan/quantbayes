# QuantBayes

# Stochastic Differential Equations Module

This module implements several stochastic differential equations (SDE) models — classical and Bayesian variants — and a neural SDE. These models are applicable in fields such as quantitative finance and physics for simulating processes subject to randomness.
<table>
  <tr>
    <td align="center">
      <img src="images/sde.png" alt="SDE" width="300"><br>
      SDE
    </td>
    <td align="center">
      <img src="images/stock_price_forecasting.png" alt="Stock Price Forecasting" width="300"><br>
      Stock Price Forecasting
    </td>
  </tr>
</table>

## 1. Geometric Brownian Motion (GBM)

GBM is widely used to model asset prices. Its dynamics are given by:

$$
dX_t = \mu X_t\,dt + \sigma X_t\,dW_t,
$$

where  
- $\mu$ is the drift,  
- $\sigma$ is the volatility, and  
- $dW_t$ represents increments of a Wiener process.

The closed-form solution is:

$$
X_t = X_0 \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right].
$$

Our implementation estimates parameters via log returns and simulates paths using the Euler-Maruyama scheme.

## 2. Heston Model

The Heston model captures stochastic volatility through two coupled SDEs:

$$
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)},\\
dv_t &= \kappa (\theta - v_t)\,dt + \sigma \sqrt{v_t}\,dW_t^{(2)},
\end{aligned}
$$

with  
- $S_t$ as the asset price,  
- $v_t$ as the instantaneous variance,  
- $\kappa$ the rate of mean reversion,  
- $\theta$ the long-term variance, and  
- $\rho$ the correlation between the two Brownian motions $dW_t^{(1)}$ and $dW_t^{(2)}$.

Simulation is performed using a correlated Euler scheme.

## 3. Merton Jump Diffusion

To model sudden price changes (jumps), Merton’s jump-diffusion model augments GBM with a jump component:

$$
dS_t = S_t \left(\mu\,dt + \sigma\,dW_t\right) + S_t (J - 1)\,dN_t,
$$

where  
- $dN_t$ is a Poisson process with intensity $\lambda$, and  
- $J$ (typically log-normally distributed) represents the jump multiplier.

This allows the model to capture both continuous fluctuations and discrete jumps.

## 4. Ornstein-Uhlenbeck Process (OU)

The OU process is a mean-reverting process described by:

$$
dX_t = \theta (\mu - X_t)\,dt + \sigma\,dW_t,
$$

where  
- $\theta$ is the speed of mean reversion,  
- $\mu$ is the long-term mean, and  
- $\sigma$ is the volatility.

We estimate parameters using a method-of-moments approach and simulate using Euler-Maruyama.

## 5. Bayesian Variants

For GBM, Heston, Merton Jump Diffusion, and OU, Bayesian versions are provided. In these models, parameters (e.g., $\mu$, $\sigma$, $\theta$) are treated as random variables with specified priors. For example, in Bayesian GBM:

- **Drift Prior:**  
  $\mu \sim \mathcal{N}(\hat{\mu}, \tau_\mu^2)$,
- **Volatility Prior:**  
  $\sigma \sim \text{LogNormal}(\log(\hat{\sigma}), \tau_\sigma)$,

The posterior inference uses MCMC (specifically the No-U-Turn Sampler, NUTS) to obtain a distribution over the parameters.

## 6. Linear Neural SDE (LSDE)

The LSDE extends the traditional SDE framework by parameterizing the drift and diffusion terms with linear neural networks. The SDE is expressed as:

$$
dX_t = f_{\text{drift}}(X_t, t)\,dt + f_{\text{diffusion}}(X_t, t)\,dW_t,
$$

where $f_{\text{drift}}$ and $f_{\text{diffusion}}$ are linear functions learned during training. This allows the model to capture more complex dynamics while retaining interpretability.



---

## Contributing

We welcome contributions from the community! Feel free to:

-   Report bugs or request features via [GitHub Issues](https://github.com/josephmargaryan/quantbayes/issues).
    
-   Submit pull requests to enhance the library.
    

## License
This project is licensed under the GPL v3. For proprietary licensing options, contact josephmargaryan@gmail.com.

## Acknowledgments

QuantBayes began as a project to advance Bayesian machine learning for academia and industry, combining rigorous research with practical tools—thanks to the bioinformatics and machine learning communities' inspiration and support.

