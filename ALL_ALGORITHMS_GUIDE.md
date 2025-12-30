# Complete ML Algorithms Guide - All 27+ Models

## Comprehensive Coverage of Machine Learning Algorithms

---

## 📊 Classification Algorithms (10 models)

### 1. **Logistic Regression**
- **Type**: Linear model
- **Best for**: Binary/multiclass classification, baseline model
- **Scaling**: Required
- **Key Hyperparameters**: C (regularization), penalty (L1/L2)
- **Pros**: Fast, interpretable, probabilistic output
- **Cons**: Linear decision boundary only

### 2. **Random Forest Classifier**
- **Type**: Ensemble (bagging)
- **Best for**: General-purpose classification, feature importance
- **Scaling**: Not required
- **Key Hyperparameters**: n_estimators, max_depth, min_samples_split
- **Pros**: Robust, handles non-linear, feature importance
- **Cons**: Can be slow, memory intensive

### 3. **Decision Tree Classifier**
- **Type**: Tree-based
- **Best for**: Interpretable models, categorical features
- **Scaling**: Not required
- **Key Hyperparameters**: max_depth, min_samples_split, criterion
- **Pros**: Highly interpretable, handles non-linear
- **Cons**: Prone to overfitting

### 4. **K-Nearest Neighbors (KNN) Classifier**
- **Type**: Instance-based
- **Best for**: Small datasets, non-linear boundaries
- **Scaling**: **Required** (distance-based)
- **Key Hyperparameters**: n_neighbors, weights, metric
- **Pros**: Simple, no training phase
- **Cons**: Slow prediction, memory intensive

### 5. **Naive Bayes (GaussianNB)**
- **Type**: Probabilistic
- **Best for**: Text classification, high-dimensional data
- **Scaling**: Optional
- **Key Hyperparameters**: var_smoothing
- **Pros**: Fast, works well with small data
- **Cons**: Assumes feature independence

### 6. **Support Vector Machine (SVM)**
- **Type**: Kernel-based
- **Best for**: Small-medium datasets, complex boundaries
- **Scaling**: **Required**
- **Key Hyperparameters**: C, kernel, gamma
- **Pros**: Effective in high dimensions, versatile kernels
- **Cons**: Slow on large datasets, memory intensive

### 7. **Gradient Boosting Machine (GBM)**
- **Type**: Ensemble (boosting)
- **Best for**: Tabular data, competitions
- **Scaling**: Not required
- **Key Hyperparameters**: n_estimators, learning_rate, max_depth
- **Pros**: High accuracy, handles mixed types
- **Cons**: Can overfit, slower training

### 8. **XGBoost Classifier**
- **Type**: Gradient boosting
- **Best for**: Structured data, competitions
- **Scaling**: Not required
- **Key Hyperparameters**: n_estimators, learning_rate, max_depth, subsample
- **Pros**: Fast, regularization, handles missing data
- **Cons**: Many hyperparameters to tune

### 9. **LightGBM Classifier**
- **Type**: Gradient boosting
- **Best for**: Large datasets, fast training
- **Scaling**: Not required
- **Key Hyperparameters**: num_leaves, learning_rate, max_depth
- **Pros**: Very fast, memory efficient
- **Cons**: Prone to overfitting on small data

### 10. **MLP Neural Network (ANN/FNN)**
- **Type**: Neural network
- **Best for**: Complex patterns, large datasets
- **Scaling**: **Required**
- **Key Hyperparameters**: hidden_layer_sizes, activation, alpha
- **Pros**: Universal approximator, flexible
- **Cons**: Requires more data, longer training

---

## 📈 Regression Algorithms (10 models)

### 1. **Linear Regression**
- **Best for**: Baseline, interpretable predictions
- **Scaling**: Recommended
- **Key Hyperparameters**: fit_intercept
- **Formula**: y = β₀ + β₁x₁ + β₂x₂ + ... + ε

### 2. **Ridge Regression (L2)**
- **Best for**: Multicollinearity, many features
- **Scaling**: **Required**
- **Key Hyperparameters**: alpha (regularization strength)
- **Regularization**: Σβ²

### 3. **Lasso Regression (L1)**
- **Best for**: Feature selection, sparse solutions
- **Scaling**: **Required**
- **Key Hyperparameters**: alpha
- **Regularization**: Σ|β|

### 4. **ElasticNet (L1 + L2)**
- **Best for**: Many correlated features
- **Scaling**: **Required**
- **Key Hyperparameters**: alpha, l1_ratio
- **Regularization**: α(ρΣ|β| + (1-ρ)Σβ²)

### 5. **Random Forest Regressor**
- **Best for**: Non-linear relationships, robustness
- **Scaling**: Not required
- **Key Hyperparameters**: n_estimators, max_depth

### 6. **Decision Tree Regressor**
- **Best for**: Interpretability, non-linear
- **Scaling**: Not required
- **Key Hyperparameters**: max_depth, min_samples_split

### 7. **K-Nearest Neighbors Regressor**
- **Best for**: Local patterns, non-parametric
- **Scaling**: **Required**
- **Key Hyperparameters**: n_neighbors, weights

### 8. **Support Vector Regressor (SVR)**
- **Best for**: Non-linear regression, small-medium data
- **Scaling**: **Required**
- **Key Hyperparameters**: C, kernel, epsilon

### 9. **Gradient Boosting Regressor**
- **Best for**: High accuracy predictions
- **Scaling**: Not required
- **Key Hyperparameters**: n_estimators, learning_rate

### 10. **MLP Regressor**
- **Best for**: Complex non-linear relationships
- **Scaling**: **Required**
- **Key Hyperparameters**: hidden_layer_sizes, activation

---

## 🔷 Clustering Algorithms (4 models)

### 1. **K-Means**
- **Type**: Centroid-based
- **Best for**: Spherical clusters, known K
- **Scaling**: **Required**
- **Key Hyperparameters**: n_clusters, init
- **Pros**: Fast, simple, scalable
- **Cons**: Need to specify K, assumes spherical clusters

### 2. **DBSCAN**
- **Type**: Density-based
- **Best for**: Arbitrary shapes, outlier detection
- **Scaling**: **Required**
- **Key Hyperparameters**: eps, min_samples
- **Pros**: No need to specify K, finds outliers
- **Cons**: Sensitive to parameters, struggles with varying densities

### 3. **Hierarchical Clustering**
- **Type**: Tree-based
- **Best for**: Dendrogram analysis, nested clusters
- **Scaling**: **Required**
- **Key Hyperparameters**: n_clusters, linkage
- **Pros**: No need to pre-specify K, visualizable
- **Cons**: Slow on large datasets O(n³)

### 4. **Gaussian Mixture Models (GMM)**
- **Type**: Probabilistic
- **Best for**: Soft clustering, overlapping clusters
- **Scaling**: **Required**
- **Key Hyperparameters**: n_components, covariance_type
- **Pros**: Probabilistic assignments, flexible shapes
- **Cons**: Sensitive to initialization, can overfit

---

## 🎨 Dimensionality Reduction (5 models)

### 1. **PCA (Principal Component Analysis)**
- **Type**: Linear
- **Best for**: Feature reduction, visualization, noise reduction
- **Scaling**: **Required**
- **Key Hyperparameters**: n_components, whiten
- **Pros**: Fast, interpretable, preserves variance
- **Cons**: Linear only, assumes orthogonal components

### 2. **LDA (Linear Discriminant Analysis)**
- **Type**: Supervised linear
- **Best for**: Classification preprocessing, class separation
- **Scaling**: **Required**
- **Key Hyperparameters**: solver, shrinkage
- **Pros**: Maximizes class separability
- **Cons**: Requires labels, max n_components = n_classes - 1

### 3. **SVD (Truncated Singular Value Decomposition)**
- **Type**: Matrix factorization
- **Best for**: Sparse matrices, text data
- **Scaling**: **Required**
- **Key Hyperparameters**: n_components, algorithm
- **Pros**: Works with sparse matrices, like PCA but no mean centering
- **Cons**: Less interpretable than PCA

### 4. **t-SNE (t-distributed Stochastic Neighbor Embedding)**
- **Type**: Non-linear manifold
- **Best for**: Visualization (2D/3D), exploring clusters
- **Scaling**: **Required**
- **Key Hyperparameters**: perplexity, learning_rate, n_iter
- **Pros**: Excellent for visualization, preserves local structure
- **Cons**: Slow, non-deterministic, can't transform new data easily

### 5. **UMAP (Uniform Manifold Approximation and Projection)**
- **Type**: Non-linear manifold
- **Best for**: Visualization, faster than t-SNE
- **Scaling**: **Required**
- **Key Hyperparameters**: n_neighbors, min_dist
- **Pros**: Faster than t-SNE, preserves global + local structure
- **Cons**: More complex hyperparameters

---

## 🧠 Deep Learning Models (CNN, RNN, LSTM)

### Note on Deep Learning Models
The notebook includes **MLP (Multi-Layer Perceptron)** which covers:
- **ANN** (Artificial Neural Network) - General term
- **FNN** (Feedforward Neural Network) - Same as MLP
- **MLP** (Multi-Layer Perceptron) - Sklearn implementation

For **CNN, RNN, LSTM**, these require specialized frameworks:

### CNN (Convolutional Neural Network)
- **Best for**: Image classification, object detection
- **Framework**: TensorFlow/Keras, PyTorch
- **Key Layers**: Conv2D, MaxPooling, Dense
- **Usage**: Computer vision tasks

### RNN (Recurrent Neural Network)
- **Best for**: Sequence data, time series
- **Framework**: TensorFlow/Keras, PyTorch
- **Key Layers**: SimpleRNN, Dense
- **Usage**: Sequential data

### LSTM (Long Short-Term Memory)
- **Best for**: Long sequences, NLP
- **Framework**: TensorFlow/Keras, PyTorch
- **Key Layers**: LSTM, Dense
- **Usage**: Text, speech, time series with long dependencies

---

## 📋 Quick Selection Guide

### Choose by Problem Type:

| Your Problem | Recommended Algorithms (in order) |
|--------------|-----------------------------------|
| **Binary Classification** | Logistic Regression → Random Forest → XGBoost → LightGBM → SVM |
| **Multiclass Classification** | Random Forest → XGBoost → LightGBM → MLP → SVM |
| **Regression** | Ridge/Lasso → Random Forest → XGBoost → LightGBM → SVR |
| **Clustering** | K-Means → DBSCAN → GMM → Hierarchical |
| **Dimensionality Reduction** | PCA → t-SNE (visualization) → UMAP → LDA (supervised) |
| **Feature Selection** | Lasso → Random Forest (importance) → LDA |
| **Interpretability** | Logistic Regression → Decision Tree → Linear Regression → Naive Bayes |
| **Speed** | Naive Bayes → Logistic Regression → K-Means → Decision Tree |
| **Accuracy (tabular)** | XGBoost → LightGBM → GBM → Random Forest |

### Choose by Dataset Size:

| Dataset Size | Best Algorithms |
|--------------|-----------------|
| **< 1K samples** | Logistic Regression, Naive Bayes, KNN, SVM |
| **1K - 10K** | Random Forest, SVM, MLP, Decision Tree |
| **10K - 100K** | Random Forest, XGBoost, GBM, LightGBM |
| **> 100K** | LightGBM, XGBoost, SGD variants, MLP |

### Choose by Feature Count:

| Features | Best Algorithms |
|----------|-----------------|
| **< 10** | Any algorithm |
| **10 - 50** | Random Forest, XGBoost, SVM, MLP |
| **50 - 200** | LightGBM, XGBoost, Ridge/Lasso (with regularization) |
| **> 200** | PCA/LDA first, then LightGBM, ElasticNet |

---

## 🔧 Scaling Requirements Summary

### **MUST Scale (distance-based):**
- KNN (both classification & regression)
- SVM/SVR
- MLP/ANN/FNN
- Logistic Regression
- Ridge/Lasso/ElasticNet
- All dimensionality reduction (PCA, LDA, SVD, t-SNE, UMAP)
- All clustering (K-Means, DBSCAN, Hierarchical, GMM)

### **No Scaling Needed (tree-based):**
- Decision Trees
- Random Forest
- XGBoost
- LightGBM
- GBM
- Naive Bayes (optional)

---

## 📊 Performance Comparison (Typical Rankings)

### Accuracy (Tabular Data):
1. XGBoost / LightGBM
2. Gradient Boosting
3. Random Forest
4. MLP
5. SVM

### Speed (Training):
1. Naive Bayes
2. Logistic Regression
3. Decision Tree
4. LightGBM
5. K-Means

### Interpretability:
1. Linear/Logistic Regression
2. Decision Tree
3. Naive Bayes
4. KNN
5. Random Forest (feature importance)

---

## 🎯 Usage in Notebook

All models are automatically configured in the notebook. Simply:

1. **Run the notebook** with your dataset
2. **Specify problem type** (or let it auto-detect)
3. **Choose tuning strategy**:
   - `RandomizedSearchCV` (default, recommended)
   - `GridSearchCV` (exhaustive)
   - `Bayesian` (Optuna)
4. **Models are automatically selected** based on problem type
5. **Best model is automatically chosen** based on validation performance

---

## 📚 Implementation Status

✅ **Fully Implemented** (ready to use):
- All 10 Classification models
- All 10 Regression models
- All 4 Clustering models
- All 5 Dimensionality Reduction models
- MLP (covers ANN/FNN/MLP)

⚠️ **Requires Additional Setup**:
- CNN, RNN, LSTM (need TensorFlow/PyTorch integration)

---

## 🚀 Next Steps

To use deep learning models (CNN, RNN, LSTM):
1. Install TensorFlow: `pip install tensorflow`
2. Use separate notebook cells for TensorFlow/Keras models
3. Follow same preprocessing pipeline
4. Adapt data format (images for CNN, sequences for RNN/LSTM)

---

**Total Algorithms Covered: 27+ models** across all ML paradigms!
