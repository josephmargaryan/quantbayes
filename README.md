# QuantBayes

QuantBayes is an advanced probabilistic machine learning library integrating **Bayesian inference, stochastic differential equations (SDEs), and deep learning**. It provides utilities for **training deterministic and probabilistic models in PyTorch (`torch.nn`) and Flax (`flax.linen`)**, supporting tasks such as classification, regression, segmentation, and generative modeling.

## Features

### **1. Deterministic & Bayesian Learning**
- Unified support for **PyTorch** and **Flax (JAX)** architectures.
- Probabilistic inference with:
  - **NUTS (No-U-Turn Sampler)**
  - **SVI (Stochastic Variational Inference)**
  - **Stein Variational Inference**
- AutoML utilities for:
  - **Binary, Multiclass Classification**
  - **Regression**
  - **Image Classification, Segmentation**

### **2. Stochastic & Probabilistic Models**
- **Stochastic Differential Equations (SDEs)**: Euler-Maruyama, Milstein schemes.
- **Bayesian Deep Learning**: GANs, VAEs, Deep Markov Models.
- **Time Series Forecasting**: Bayesian LSTMs, Transformers, Gaussian Processes.

### **3. Efficient Computation**
- **FFT-based Bayesian Layers** for fast inference.
- **Quantitative Finance Applications**: Volatility modeling, probabilistic forecasting.
- **Visualization**: Calibration plots, posterior distributions, uncertainty estimation.

---

## Installation

QuantBayes can be cloned and, in the future,e installed using pip:
`git clone https://github.com/josephmargaryan/quantbayes.git`
## Getting Started

### Example: Building a Custom Bayesian Neural Network

    import jax
    import jax.numpy as jnp
    from quantbayes.nn import Linear, BaseModel, FFTLinear
    
    class CustomBNN(BaseModel):
        def __init__(self):
            super().__init__(task_type="regression", method="nuts")    # takes method: str ("nuts", "svi", "steinvi") 
        def __call__(self, x, y=None):                                 # task_type: str ("regression", "multiclass", "binary", "image_segmentation", "image_classification")
            n, in_features = x.shape
            x = FFTLinear(in_features, name="fft layer")(x)
            x = jax.nn.tanh(x)
            mean = Linear(in_features, 1)(x)
            mean = mean.squeeze()
            sigma = numpyro.sample("sigma", dist.Exponential(1.0))
            numpyro.deterministic("logits", mean.squeeze()) 
            numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)
            
          
    # Initialize train, predict and visualize performance
    key = jax.random.key(0)
    model = CustomBNN()
    model.compile()
    model.fit(X, y, key, num_warmups=500, num_samples=1000, num_chains=1)
    preds = model.predict(X_test, key) # Full posterior preds (logits)
    model.visualize(X, y, feature_index=0) 
    
    ### Get Pac-Bayes Bound
    from bnn.utils import BayesianAnalysis
    bound = BayesianAnalysis(len(X_train), 
        delta=0.05, 
        task_type="regression", 
        inference_type="mcmc", 
        posterior_samples=model.get_samples,
    )
    bound.compute_pac_bayesian_bound(preds, y)


### Example: Using Prebuilt AutoML Models

    from quantbayes.bnn import DenseBinaryiSteinVI
    
    # Initialize a prebuilt model for binary classification
    model = DenseBinaryiSteinVI(num_particles=10, 
                    model_type="deep", 
                    hidden_size=3, 
                    num_steps=100
                    )
    
    # Train and predict
    model.fit(X_train, y_train, key)
    preds = model.predict(X_test, key)
    model.visualize(X=X_test, y=y_test, resolution=100, features=(0, 1))



### Highlights

-   **Bayesian Neural Networks:** Fine-tuned pipelines for uncertainty estimation and robust predictions.
    
-   **FFT Layers:** Efficient, frequency-domain computations for fast and scalable modeling.
    
-   **Visualization Tools:** Calibration plots, uncertainty visualization, and more to evaluate models effectively.

![Descriptive Caption](path_to_image.png)

### Reinforcement Learning
- **Classification**: Predictions sampled from a categorical distribution, optimizing rewards based on correctness.
- **Regression**: Probabilistic outputs modeled via normal distributions, with rewards linked to error minimization.
- **Image Segmentation**: Pixel-wise classification treated as an action space for structured decision-making.
- **Time-Series Forecasting**: LSTM-based policies leverage temporal dependencies and predict directional movements.


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

QuantBayes began as a project to advance Bayesian machine learning for academia and industry, combining rigorous research with practical tools—special thanks to the bioinformatics and machine learning communities for their inspiration and support.

