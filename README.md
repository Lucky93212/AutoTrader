# Multi-Level Ensemble Stock Prediction System

This repository contains my implementation of an advanced algorithmic trading system that combines multiple machine learning methodologies to predict individual stock returns. The system is based on and extends the work from [k-dickinson/quant-simulations-and-risk](https://github.com/k-dickinson/quant-simulations-and-risk), incorporating state-of-the-art ensemble methods and risk management techniques.

## Core Philosophy: Independent Stock Modeling

I've designed this system to treat each stock as an independent time series prediction problem allowing for the following advantages:

1. **Reduced Model Risk**: By avoiding correlation assumptions, the system is less susceptible to correlation breakdown during market stress
2. **Scalability**: Each stock model can be trained independently, allowing for parallel processing and easier system scaling
3. **Robustness**: Individual models capture stock-specific dynamics that might be obscured in factor-based approaches

*Note: Future iterations will incorporate cross-asset relationships and regime-dependent correlations as outlined in the risk management framework.*

## Machine Learning Architecture

### Multi-Level Ensemble Stacking

The strategy itself lies in the three-level ensemble architecture that combines diverse learning paradigms. While this is, in no way, revolutionary, it provides consistency as opposed to more subjective approaches:

#### Level 1: Base Model Ensemble
The first level employs heterogeneous base models to capture different aspects of market dynamics:

**Neural Network (MLP Regressor)**
- **Architecture**: Deep feed-forward network with batch normalization and dropout
- **Activation Functions**: Alternating ReLU and LeakyReLU to prevent vanishing gradients
- **Regularization**: L1 penalty (λ = 1×10⁻⁵) + gradient clipping + early stopping
- **Optimization**: AdamW with cosine annealing warm restarts

The loss function combines MSE with L1 regularization: $L = \text{MSE}(y, \hat{y}) + \lambda \lVert \theta \rVert_1$. Where θ represents the model parameters and λ controls sparsity.

**Gradient Boosting Models (XGBoost & LightGBM)**
- **XGBoost**: GPU-accelerated tree boosting with sophisticated regularization
- **LightGBM**: Leaf-wise tree growth optimized for efficiency
- **Hyperparameters**: Conservative learning rates (0.01) with high iteration counts for stability

**Boosted Residual Models**
Here I trained traditional ML models (SVR, Random Forest, Lasso) as base predictors, then use XGBoost to learn the residual patterns: $\hat{y}{\text{final}} = f_{\text{base}}(X) + f_{\text{boost}}(X, \text{residuals})$. This captures both linear and non-linear components of the prediction problem.

**AutoML (AutoGluon)**
- Automated hyperparameter optimization across multiple model families
- Stacked generalization with automatic feature selection
- Provides robust baseline predictions with minimal manual tuning

**TabPFN (Transformer-based)**
- Prior-fitted network leveraging transformer architecture
- Particularly effective for tabular data with complex feature interactions
- Serves as a strong non-parametric baseline

#### Level 2: Meta-Learning
The second level trains meta-models on the predictions from Level 1:

**XGBoost Meta-Learner**
- Learns optimal combination weights for base model predictions
- Captures non-linear interactions between base model outputs
- Regularized to prevent overfitting to training ensemble diversity

**Neural Network Meta-Learner**
- Smaller architecture (fewer layers) to avoid overfitting
- Learns complex combination rules between base predictions
- Dropout and early stopping for regularization

#### Level 3: Final Ensemble
The third level employs Ridge regression to learn optimal linear weights: $\hat{y}{\text{final}} = \sum_{i} w_i \cdot \hat{y}_i^{(2)} + \varepsilon$. Where ŷᵢ⁽²⁾ are Level 2 predictions and wᵢ are learned weights with L2 regularization.

### Financial Theory Foundation

#### Technical Analysis Integration
The feature engineering incorporates established technical analysis principles:

**Momentum Indicators**
- RSI (Relative Strength Index): Captures overbought/oversold conditions
- Rate of Change (ROC): Multi-timeframe momentum assessment
- MACD: Trend-following momentum oscillator

**Volatility Measures**
- Bollinger Bands: Statistical volatility bands for mean reversion signals
- Average True Range: Volatility-adjusted position sizing
- Rolling standard deviation: Multi-horizon volatility estimation

**Volume Analysis**
- On-Balance Volume (OBV): Price-volume confirmation
- Volume Rate of Change: Institutional activity detection
- Volume-weighted indicators: Smart money tracking

#### Efficient Market Hypothesis Considerations
While respecting EMH principles, the system exploits several well-documented market inefficiencies:

1. **Momentum Effects**: Short-term continuation of price trends
2. **Mean Reversion**: Long-term price normalization
3. **Volatility Clustering**: GARCH-like volatility patterns
4. **Microstructure Effects**: Intraday trading patterns

#### Modern Portfolio Theory Extensions
The risk management framework extends traditional MPT:

**Value at Risk (VaR) Calculation**
Using Monte Carlo simulation with Geometric Brownian Motion: $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$, where:

- μ = drift parameter (estimated from historical returns)
- σ = volatility parameter (estimated from historical volatility)
- dW_t = Wiener process (random walk component)

**Risk-Adjusted Position Sizing**
Position weights are calculated using a confidence-weighted approach: $w_i = \frac{ \left| r_i \right| \cdot c_i^\alpha }{ \sum_j \left| r_j \right| \cdot c_j^\alpha } \cdot A$, where:

- r_i = predicted return for asset i
- c_i = prediction confidence for asset i
- α = confidence exponent (1.5 for non-linear emphasis)
- A = total allocation constraint (90%)

## Risk Management Framework

### Monte Carlo Simulation
The system employs sophisticated Monte Carlo methods for portfolio risk assessment:

**Geometric Brownian Motion Implementation**

$S(t) = S_0 \cdot \exp\left( \left( \mu - \frac{\sigma^2}{2} \right) t + \sigma \sqrt{t} \cdot Z \right)$

Where Z ~ N(0,1) represents random market shocks.

**Risk Metrics Calculation**
- **95% Value at Risk**: 5th percentile of return distribution
- **Expected Shortfall**: Mean of worst 5% outcomes
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Probability of Loss**: Frequency of negative returns

### Dynamic Position Sizing
The position sizing algorithm incorporates multiple risk factors:

1. **Prediction Confidence**: Higher confidence → larger positions
2. **Volatility Adjustment**: Higher volatility → smaller positions
3. **Correlation Limits**: Reduces exposure to highly correlated positions
4. **Concentration Limits**: Maximum 10% per single position

## Implementation Details

### Independent Stock Processing
Each stock is processed independently with its own:
- Feature engineering pipeline
- Model ensemble training
- Hyperparameter optimization
- Validation framework

This approach ensures that:
- Model failures are isolated
- Stock-specific patterns are captured
- System scales linearly with universe size

### GPU Acceleration
The system leverages GPU acceleration where available:
- PyTorch neural networks on CUDA
- XGBoost tree methods with GPU histograms
- LightGBM GPU training
- Parallel Monte Carlo simulations

### Model Persistence
Trained models are serialized with:
- Complete ensemble architecture
- Training metadata and statistics
- Feature engineering parameters
- Risk model coefficients

## Performance Considerations

### Bias-Variance Tradeoff
The multi-level ensemble addresses the bias-variance tradeoff by:
- **Reducing Bias**: Multiple diverse base models
- **Controlling Variance**: Regularization at each level
- **Optimal Combination**: Meta-learning for weight optimization

### Overfitting Prevention
Multiple regularization techniques prevent overfitting:
1. **Cross-validation**: Time-series aware validation splits
2. **Early stopping**: Based on validation loss plateaus
3. **Dropout**: Stochastic regularization in neural networks
4. **L1/L2 penalties**: Parameter shrinkage
5. **Ensemble diversity**: Decorrelated base models

## Future Enhancements

1. **Cross-Asset Modeling**: Incorporate sector and market regime information
2. **Alternative Data**: Integration of sentiment, news, and satellite data
3. **Reinforcement Learning**: Dynamic strategy adaptation
4. **Options Integration**: Volatility surface modeling for options strategies
5. **High-Frequency Components**: Intraday pattern recognition

## Mathematical Notation Summary

Key equations used throughout the system:

**Ensemble Prediction**:
$$\hat{y} = \sum_{i=1}^{M} w_i \cdot f_i(X)$$

**Risk-Adjusted Return**:
$$R_{adj} = \frac{E[R] - R_f}{\sigma_p}$$

**Position Weight**:
$$w_i = \frac{|r_i| \cdot c_i^{\alpha}}{\sum_{j=1}^{N} |r_j| \cdot c_j^{\alpha}} \cdot A$$

**Value at Risk**:
$$VaR_{\alpha} = -\inf\{x \in \mathbb{R} : P(X \leq x) \geq \alpha\}$$

This system represents my approach to quantitative trading that combines machine learning techniques with established financial theory, designed to generate consistent risk-adjusted returns through independent stock modeling and advanced ensemble methods.
