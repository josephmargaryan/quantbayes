
# QuantBayes

Welcome to **QuantBayes**, an advanced machine learning library designed to empower researchers, data scientists, and developers to build, train, and evaluate sophisticated probabilistic and deterministic models. This library integrates Bayesian machine learning, stochastic processes, and deep learning under a unified framework, providing modular tools for cutting-edge research and practical applications.

## Features

-   **Modular Design:** Inspired by PyTorch's `torch.nn`, QuantBayes offers highly modular components such as `Linear`, `Conv2d`, `TransformerEncoder`, and more, allowing users to define custom architectures with ease.
    
-   **Probabilistic Inference:** Support for Bayesian inference methods, including:
    
    -   NUTS (No-U-Turn Sampler)
        
    -   SVI (Stochastic Variational Inference)
        
    -   SteinVI (Stein Variational Inference)
        
-   **FFT-Accelerated Models:** Specialized FFT-based layers for efficient computation in Bayesian machine learning.
    
-   **Automated Pipelines:** Prebuilt AutoML classes for classification, regression, and binary tasks, enabling end-to-end training and evaluation.
    
-   **Stochastic Differential Equations:** Implementations and tools for solving SDEs, including applications in generative adversarial networks (GANs).
    
-   **Quantitative Finance Tools:** Models like stochastic volatility and ensemble methods for time series forecasting and financial analysis.
    
-   **Visualization and Analysis:** Built-in utilities for calibration, uncertainty quantification, and PAC-Bayesian bound analysis.
    

## Installation

QuantBayes can be cloned and, in the future,e installed using pip:
`git clone https://github.com/josephmargaryan/quantbayes.git`
## Getting Started

### Example: Building a Custom Bayesian Neural Network

    import jax
    import jax.numpy as jnp
    from quantbayes.nn import Linear, BaseModel, FFTLinear
    
    class CustomBNN(BaseModel):
        def __init__(self, input_dim):
            super().__init__(task_type="regression", method="nuts")
            self.linear1 = FFTLinear(input_dim, name="fft layer")
            self.linear2 = Linear(input_dim, 1, name="layer2")
    
        def forward(self, x):
            x = jax.nn.tanh(self.linear1(x))
            return self.linear2(x)
    
    
          
    # Initialize train, predict and visualize performance
    key = jax.random.PRNGKey(0)
    model = CustomBNN(input_dim=X.shape[1])
    model.compile()
    model.fit(X, y, key, num_warmups=500, num_samples=1000, num_chains=1)
    preds = model.predict(X_test, key) # Full posterior preds (logits)
    model.visualize(X, y, feature_index=0) 
    
    ### Get Pac-Bayes Bound
    from bnn.utils import BayesianAnalysis
    bound = BayesianAnalysis(len(X_train), delta=0.05, task_type="regression", inference_type="mcmc", posterior_samples=model.get_samples,
    )
    bound.compute_pac_bayesian_bound(preds, y)


### Example: Using Prebuilt AutoML Models

    from quantbayes.bnn import DenseBinaryiSteinVI
    
    # Initialize a prebuilt model for binary classification
    model = DenseBinaryiSteinVI(num_particles=10, model_type="deep", hidden_size=3, num_steps=100)
    
    # Train and predict
    model.fit(X_train, y_train, key)
    preds = model.predict(X_test, key)
    model.visualize(X=X_test, y=y_test, resolution=100, features=(0, 1))


## Components

### Core Modules

-   `nn`: Includes foundational building blocks such as `Linear`, `Conv2d`, `TransformerEncoder`, and more.
    
-   `bnn`: Provides AutoML classes for Bayesian neural networks with different inference techniques.
    
-   `sde`: Tools for solving stochastic differential equations, including GANs and differential equation solvers.
    
-   `forecast`: Models and utilities for time series forecasting, including preprocessor modules.
    
-   `utils`: Helper functions for entropy calculations, PAC-Bayesian bounds, and data visualization.
### Highlights

-   **Bayesian Neural Networks:** Fine-tuned pipelines for uncertainty estimation and robust predictions.
    
-   **FFT Layers:** Efficient, frequency-domain computations for fast and scalable modeling.
    
-   **Visualization Tools:** Calibration plots, uncertainty visualization, and more to evaluate models effectively.

### Pac Analysis Bound 
To evaluate the generalization capabilities of Bayesian Neural Networks (BNNs), we compute the PAC-Bayesian bound:

  

1. **Empirical Risk**: The average loss across posterior samples:

$$
\hat{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}_{h \sim Q} \left[ \ell(h(x_i), y_i) \right],
$$

where $\ell$ is the task-specific loss function, and $(x_i, y_i)$ are the data points.

2. **KL Divergence**: Measures the complexity of the posterior \(Q\) relative to the prior \(P\):


$$
\hat{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}_{h \sim Q} \left[ \ell(h(x_i), y_i) \right],
$$

where $\mu_j, \sigma_j$ are the posterior mean and standard deviation, and $\mu_{\text{prior}}, \sigma_{\text{prior}}$ are the prior parameters.

  

3. **PAC-Bayesian Bound**: Combines empirical risk and complexity:


$$
L(Q) \leq \hat{L}(Q) + \sqrt{\frac{D_{KL}(Q \| P) + \ln(1 / \delta)}{2n}}
$$

where $L(Q)$ is the true risk, $\delta$ is the confidence level, and $n$ is the number of training samples.

## Why QuantBayes?

QuantBayes stands out for its:

-   **Breadth and Depth:** Combining Bayesian, stochastic, and deep learning techniques in one library.
    
-   **Modularity:** Flexible components for custom architectures or quick-start prebuilt pipelines.
    
-   **Efficiency:** Optimized with FFT and probabilistic methods for cutting-edge performance.
    

## Contributing

We welcome contributions from the community! Feel free to:

-   Report bugs or request features via [GitHub Issues](https://github.com/josephmargaryan/quantbayes/issues).
    
-   Submit pull requests to enhance the library.
    

## License
This project is licensed under the GPL v3. For proprietary licensing options, contact josephmargaryan@gmail.com.

## Acknowledgments

QuantBayes began as a project to advance Bayesian machine learning for academia and industry, combining rigorous research with practical tools. Special thanks to the bioinformatics and machine learning communities for their inspiration and support.

