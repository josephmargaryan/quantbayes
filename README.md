# ChronosForge

ChronosForge is a comprehensive library bridging time-series forecasting, probabilistic modeling, and cutting-edge machine-learning techniques. Inspired by the Greek god of time, **Chronos**, this library symbolizes precision, inevitability, and the exploration of truth through data.

---

## 🌌 Philosophical Inspiration

In Greek mythology, Chronos embodies time and its infinite nature. This library reflects that spirit by crafting tools that transcend time—forecasting the future, analyzing the present, and uncovering the hidden patterns of the past. Whether solving stochastic equations, creating neural transport models, or designing probabilistic machine learning algorithms, **ChronosForge** equips you with the tools to forge solutions in the ever-flowing river of time.

---

## 🚀 Features

ChronosForge includes a variety of modules and algorithms spanning multiple disciplines:

### **Time Forecasting**
- **ARCH**: Analyze volatility and financial time series.
- **LSTM, Transformers**: Deep learning models for sequence prediction.
- **Stochastic Volatility Models**: Predict uncertainty in financial markets.
- **Causal Discovery**: Uncover causal relationships in time-series data.

### **Probabilistic Machine Learning**
- **Gaussian Processes (GP)**: Flexible models for uncertainty quantification.
- **Variational Inference (VI)**: Scalable Bayesian inference techniques.
- **Markov Models (HMM, DMM)**: For sequential data analysis.
- **Monte Carlo Methods**: No U-turn sampler to estimate the posterior distribution![CodeCogsEqn](https://github.com/user-attachments/assets/e6ca9fe8-3f34-4359-8936-459e4b0382f8)


### **Optimization and Theoretical Bounds**
- **Constrained Lagrangian Optimization**: Solve constrained optimization problems.
![CodeCogsEqn](https://github.com/user-attachments/assets/f38dd6cf-7fc1-4ffe-89b7-54a6d8ba44ae)

- **Theoretical Bounds**: Analyze generalization using PAC-Bayes, VC dimensions, and Hoeffding inequalities.

### **Deep Learning Applications**
- **Long Sequence Classification**: Handle document-level tasks with transformers and hierarchical models.
- **Temporal Fusion Transformer (TFT)**: Integrate time-dependent and contextual data.
- **Reinforcement Learning**: Optimize decision-making in dynamic environments.

### **FFT Circulant Applications**
The **FFT Circulant Modules** in ChronosForge bring groundbreaking efficiency to matrix operations, reducing the time complexity from \(O(n^2)\) to \(O(n \log n)\). These modules excel in:
- Capturing periodicity in features for improved modeling accuracy.
- Efficiently handling large-scale matrix multiplications in probabilistic and time-series models, where the features express periodicity.


![CodeCogsEqn](https://github.com/user-attachments/assets/07c0c15d-5027-4899-8732-4e56a309542c)


![FFT Circulant Example](images/fft_circulant_example.png)

> **Figure**: Comparison of \(O(n^2)\) vs \(O(n \log n)\) matrix multiplications, highlighting the power of FFT Circulant techniques.

---

## 📊 Visualizing the Power of ChronosForge

### **1. Time-Series Sentiment Analysis**
ChronosForge provides sentiment analysis tools to evaluate market trends. Below is a 3D scatter plot of predicted sentiments for different companies.

![3D Scatter Plot of Sentiments](images/3d_scatter_plot_sentiments.png)

---

### **2. Word Clouds for Sentiment Classes**
ChronosForge visualizes sentiment-driven keywords for better interpretability. Here’s a word cloud generated for **Sentiment Class 1**.

![Word Cloud for Sentiment Class 1](images/word_cloud_sentiment_1.png)

---

### **3. Stock Price Forecasting**
ChronosForge's stochastic models can predict stock prices with high accuracy. Below is a comparison of **actual** vs. **predicted** stock prices for Novo Nordisk using ensemble techniques.

![Actual vs. Predicted Stock Prices](images/stock_price_forecasting.png)

---

### **4. Brownian Motion, Heston Models and Jump Diffusion**
ChronosForge incorporates advanced stochastic models to simulate and analyze financial markets with greater realism:

#### **Brownian Motion**
Brownian motion forms the backbone of stochastic processes in finance, modeling the random behavior of stock prices:

![CodeCogsEqn](https://github.com/user-attachments/assets/19ba8d69-d7c0-47e9-b9b6-c0a9b799bd68)

where:
- \(S_t\): Stock price at time \(t\).
- \(\mu\): Drift (average return).
- \(\sigma\): Volatility.
- \(W_t\): Wiener process (standard Brownian motion).

#### **Heston Model**
The Heston model extends Brownian motion by introducing stochastic volatility:
![CodeCogsEqn](https://github.com/user-attachments/assets/cb49ab9e-5946-46e8-8581-5aff76025242)
where:
- \(v_t\): Variance (stochastic volatility).
- \(\kappa\): Mean-reversion rate.
- \(\theta\): Long-term variance.
- \(\xi\): Volatility of variance.
- \(W_t\), \(Z_t\): Correlated Wiener processes.

#### **Jump Diffusion**
Jump diffusion adds discrete jumps to Brownian motion, modeling sudden market movements:
\[
![CodeCogsEqn](https://github.com/user-attachments/assets/309650d1-a11d-4139-b3c0-6f5b2c58a9a3)
\]
where:
- \(J_t = \sum_{i=1}^{N_t} Y_i\), with \(N_t\) being a Poisson process and \(Y_i\) jump sizes.

> **Visualization**:

![Jump Diffusion Simulation](images/jump_diffusion_simulation.png)

---

## 📁 Directory Structure

```
ChronosForge/
├── time_forecasting/         # Modules for forecasting and time-series analysis
├── probabilistic_ml/         # Probabilistic and Bayesian methods
├── optimization/             # Optimization algorithms and theoretical bounds
├── inference/                # MCMC, variational inference, and related topics
├── neural_models/            # LSTMs, Transformers, and TCN implementations
├── structural_bioinformatics # Bioinformatics-focused tools
├── data_utils/               # Data preprocessing and augmentation tools
├── README.md                 # Library overview
└── requirements.txt          # Python dependencies
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/chronosforge.git
cd chronosforge
pip install -r requirements.txt
```

---

## 🔥 Quick Start

Here's an example of using ChronosForge for time-series forecasting with a transformer model:

```python
from chronosforge.time_forecasting.transformer import TimeSeriesTransformer
from chronosforge.utils.data_loader import load_time_series

# Load time-series data
data = load_time_series("path/to/dataset.csv")

# Initialize and train the transformer model
model = TimeSeriesTransformer(input_dim=10, hidden_dim=128, num_heads=4)
model.train(data, epochs=50, learning_rate=1e-4)

# Make predictions
predictions = model.predict(data.test)
```

---

## 🧪 Contributing

We welcome contributions! To get started:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ❤️ Acknowledgments

Special thanks to the developers and researchers who inspired this project. ChronosForge aims to provide tools for modern research and practical applications in the vast field of machine learning and probabilistic inference.
