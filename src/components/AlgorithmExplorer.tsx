import React, { useState } from 'react';
import { Search, Filter, TrendingUp, Zap, Brain, Target, Layers, Eye, ArrowLeft } from 'lucide-react';

interface Algorithm {
  id: string;
  name: string;
  category: 'Classification' | 'Regression' | 'Clustering' | 'Dimensionality Reduction';
  description: string;
  bestFor: string[];
  scalingRequired: boolean;
  speed: 1 | 2 | 3 | 4 | 5;
  accuracy: 1 | 2 | 3 | 4 | 5;
  interpretability: 1 | 2 | 3 | 4 | 5;
  hyperparameters: string[];
  pros: string[];
  cons: string[];
}

const algorithms: Algorithm[] = [
  // Classification Algorithms
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    category: 'Classification',
    description: 'Linear model for binary/multiclass classification with probabilistic output',
    bestFor: ['Baseline models', 'Interpretable predictions', 'Binary classification'],
    scalingRequired: true,
    speed: 5,
    accuracy: 3,
    interpretability: 5,
    hyperparameters: ['C (regularization)', 'penalty (L1/L2)', 'solver'],
    pros: ['Fast training', 'Interpretable coefficients', 'Probabilistic output'],
    cons: ['Linear decision boundary only', 'Assumes linear relationships']
  },
  {
    id: 'random-forest-clf',
    name: 'Random Forest Classifier',
    category: 'Classification',
    description: 'Ensemble of decision trees using bagging for robust predictions',
    bestFor: ['General-purpose classification', 'Feature importance', 'Non-linear patterns'],
    scalingRequired: false,
    speed: 3,
    accuracy: 4,
    interpretability: 3,
    hyperparameters: ['n_estimators', 'max_depth', 'min_samples_split', 'max_features'],
    pros: ['Handles non-linearity well', 'Feature importance', 'Robust to outliers'],
    cons: ['Can be memory intensive', 'Slower predictions', 'Can overfit']
  },
  {
    id: 'decision-tree-clf',
    name: 'Decision Tree Classifier',
    category: 'Classification',
    description: 'Tree-based model that splits data based on feature values',
    bestFor: ['Interpretable models', 'Categorical features', 'Quick prototyping'],
    scalingRequired: false,
    speed: 4,
    accuracy: 3,
    interpretability: 5,
    hyperparameters: ['max_depth', 'min_samples_split', 'criterion', 'max_features'],
    pros: ['Highly interpretable', 'Fast training', 'Handles mixed data types'],
    cons: ['Prone to overfitting', 'Unstable', 'Biased with imbalanced data']
  },
  {
    id: 'knn-clf',
    name: 'K-Nearest Neighbors',
    category: 'Classification',
    description: 'Instance-based learning using distance metrics',
    bestFor: ['Small datasets', 'Non-linear boundaries', 'Pattern recognition'],
    scalingRequired: true,
    speed: 2,
    accuracy: 3,
    interpretability: 4,
    hyperparameters: ['n_neighbors', 'weights', 'metric', 'algorithm'],
    pros: ['No training phase', 'Simple concept', 'Effective for small data'],
    cons: ['Slow predictions', 'Memory intensive', 'Sensitive to irrelevant features']
  },
  {
    id: 'naive-bayes',
    name: 'Naive Bayes',
    category: 'Classification',
    description: 'Probabilistic classifier based on Bayes theorem',
    bestFor: ['Text classification', 'High-dimensional data', 'Real-time predictions'],
    scalingRequired: false,
    speed: 5,
    accuracy: 3,
    interpretability: 4,
    hyperparameters: ['var_smoothing', 'priors'],
    pros: ['Very fast', 'Works well with small data', 'Good for text'],
    cons: ['Assumes feature independence', 'Sensitive to feature correlations']
  },
  {
    id: 'svm-clf',
    name: 'Support Vector Machine (SVM)',
    category: 'Classification',
    description: 'Finds optimal hyperplane to separate classes',
    bestFor: ['Small-medium datasets', 'High-dimensional spaces', 'Complex boundaries'],
    scalingRequired: true,
    speed: 2,
    accuracy: 4,
    interpretability: 2,
    hyperparameters: ['C', 'kernel', 'gamma', 'degree'],
    pros: ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
    cons: ['Slow on large datasets', 'Sensitive to scaling', 'Hard to interpret']
  },
  {
    id: 'gradient-boosting-clf',
    name: 'Gradient Boosting (GBM)',
    category: 'Classification',
    description: 'Sequential ensemble that corrects previous errors',
    bestFor: ['Tabular data', 'Competitions', 'High accuracy needs'],
    scalingRequired: false,
    speed: 3,
    accuracy: 5,
    interpretability: 2,
    hyperparameters: ['n_estimators', 'learning_rate', 'max_depth', 'subsample'],
    pros: ['High accuracy', 'Feature importance', 'Handles missing data'],
    cons: ['Can overfit', 'Slower training', 'Many hyperparameters']
  },
  {
    id: 'xgboost-clf',
    name: 'XGBoost Classifier',
    category: 'Classification',
    description: 'Optimized gradient boosting with regularization',
    bestFor: ['Structured/tabular data', 'Kaggle competitions', 'Production systems'],
    scalingRequired: false,
    speed: 4,
    accuracy: 5,
    interpretability: 2,
    hyperparameters: ['n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree', 'gamma'],
    pros: ['Very high accuracy', 'Fast', 'Built-in regularization', 'Handles missing data'],
    cons: ['Many hyperparameters', 'Can overfit', 'Less interpretable']
  },
  {
    id: 'lightgbm-clf',
    name: 'LightGBM Classifier',
    category: 'Classification',
    description: 'Gradient boosting optimized for speed and memory',
    bestFor: ['Large datasets', 'Fast training needs', 'Memory constraints'],
    scalingRequired: false,
    speed: 5,
    accuracy: 5,
    interpretability: 2,
    hyperparameters: ['num_leaves', 'learning_rate', 'max_depth', 'min_child_samples'],
    pros: ['Very fast', 'Memory efficient', 'High accuracy', 'Handles categorical'],
    cons: ['Can overfit on small data', 'Complex hyperparameters']
  },
  {
    id: 'mlp-clf',
    name: 'MLP Neural Network (ANN/FNN)',
    category: 'Classification',
    description: 'Multi-layer perceptron with feedforward architecture',
    bestFor: ['Complex patterns', 'Large datasets', 'Non-linear relationships'],
    scalingRequired: true,
    speed: 2,
    accuracy: 4,
    interpretability: 1,
    hyperparameters: ['hidden_layer_sizes', 'activation', 'alpha', 'learning_rate', 'solver'],
    pros: ['Universal approximator', 'Flexible architecture', 'Handles complexity'],
    cons: ['Needs more data', 'Slower training', 'Hard to interpret', 'Requires tuning']
  },

  // Regression Algorithms
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    category: 'Regression',
    description: 'Ordinary Least Squares regression for continuous targets',
    bestFor: ['Baseline models', 'Linear relationships', 'Quick predictions'],
    scalingRequired: true,
    speed: 5,
    accuracy: 3,
    interpretability: 5,
    hyperparameters: ['fit_intercept', 'normalize'],
    pros: ['Very fast', 'Highly interpretable', 'Simple to implement'],
    cons: ['Only linear relationships', 'Sensitive to outliers', 'Assumes normality']
  },
  {
    id: 'ridge-regression',
    name: 'Ridge Regression (L2)',
    category: 'Regression',
    description: 'Linear regression with L2 regularization',
    bestFor: ['Multicollinearity', 'Many features', 'Preventing overfitting'],
    scalingRequired: true,
    speed: 5,
    accuracy: 3,
    interpretability: 4,
    hyperparameters: ['alpha', 'solver'],
    pros: ['Handles multicollinearity', 'Reduces overfitting', 'Stable'],
    cons: ['Linear only', 'Keeps all features', 'Requires scaling']
  },
  {
    id: 'lasso-regression',
    name: 'Lasso Regression (L1)',
    category: 'Regression',
    description: 'Linear regression with L1 regularization for feature selection',
    bestFor: ['Feature selection', 'Sparse solutions', 'High-dimensional data'],
    scalingRequired: true,
    speed: 4,
    accuracy: 3,
    interpretability: 5,
    hyperparameters: ['alpha', 'selection'],
    pros: ['Automatic feature selection', 'Interpretable', 'Sparse solutions'],
    cons: ['Linear only', 'Unstable with correlated features']
  },
  {
    id: 'elasticnet',
    name: 'ElasticNet (L1+L2)',
    category: 'Regression',
    description: 'Combines L1 and L2 regularization',
    bestFor: ['Many correlated features', 'Feature selection', 'Robust models'],
    scalingRequired: true,
    speed: 4,
    accuracy: 3,
    interpretability: 4,
    hyperparameters: ['alpha', 'l1_ratio', 'selection'],
    pros: ['Combines Ridge & Lasso benefits', 'Handles correlations', 'Feature selection'],
    cons: ['More hyperparameters', 'Linear only']
  },
  {
    id: 'random-forest-reg',
    name: 'Random Forest Regressor',
    category: 'Regression',
    description: 'Ensemble of regression trees',
    bestFor: ['Non-linear relationships', 'Feature importance', 'Robust predictions'],
    scalingRequired: false,
    speed: 3,
    accuracy: 4,
    interpretability: 3,
    hyperparameters: ['n_estimators', 'max_depth', 'min_samples_split', 'max_features'],
    pros: ['Handles non-linearity', 'Feature importance', 'Robust'],
    cons: ['Memory intensive', 'Can overfit']
  },
  {
    id: 'svr',
    name: 'Support Vector Regressor (SVR)',
    category: 'Regression',
    description: 'SVM adapted for regression tasks',
    bestFor: ['Non-linear regression', 'Small-medium data', 'Robust predictions'],
    scalingRequired: true,
    speed: 2,
    accuracy: 4,
    interpretability: 2,
    hyperparameters: ['C', 'kernel', 'epsilon', 'gamma'],
    pros: ['Handles non-linearity', 'Robust to outliers', 'Effective in high dims'],
    cons: ['Slow on large data', 'Many hyperparameters', 'Hard to interpret']
  },
  {
    id: 'xgboost-reg',
    name: 'XGBoost Regressor',
    category: 'Regression',
    description: 'Gradient boosting for regression',
    bestFor: ['High accuracy', 'Competitions', 'Structured data'],
    scalingRequired: false,
    speed: 4,
    accuracy: 5,
    interpretability: 2,
    hyperparameters: ['n_estimators', 'learning_rate', 'max_depth', 'subsample'],
    pros: ['Very high accuracy', 'Fast', 'Handles missing data'],
    cons: ['Many hyperparameters', 'Can overfit']
  },
  {
    id: 'mlp-reg',
    name: 'MLP Regressor',
    category: 'Regression',
    description: 'Neural network for regression',
    bestFor: ['Complex non-linear patterns', 'Large datasets'],
    scalingRequired: true,
    speed: 2,
    accuracy: 4,
    interpretability: 1,
    hyperparameters: ['hidden_layer_sizes', 'activation', 'alpha', 'learning_rate'],
    pros: ['Universal approximator', 'Flexible', 'Handles complexity'],
    cons: ['Needs more data', 'Slow training', 'Hard to interpret']
  },

  // Clustering Algorithms
  {
    id: 'kmeans',
    name: 'K-Means',
    category: 'Clustering',
    description: 'Centroid-based clustering algorithm',
    bestFor: ['Spherical clusters', 'Known number of clusters', 'Fast clustering'],
    scalingRequired: true,
    speed: 5,
    accuracy: 3,
    interpretability: 4,
    hyperparameters: ['n_clusters', 'init', 'max_iter', 'n_init'],
    pros: ['Very fast', 'Simple', 'Scalable'],
    cons: ['Need to specify K', 'Assumes spherical clusters', 'Sensitive to initialization']
  },
  {
    id: 'dbscan',
    name: 'DBSCAN',
    category: 'Clustering',
    description: 'Density-based clustering with outlier detection',
    bestFor: ['Arbitrary shapes', 'Outlier detection', 'Unknown cluster count'],
    scalingRequired: true,
    speed: 3,
    accuracy: 4,
    interpretability: 3,
    hyperparameters: ['eps', 'min_samples', 'metric'],
    pros: ['No need to specify K', 'Finds outliers', 'Arbitrary shapes'],
    cons: ['Sensitive to parameters', 'Struggles with varying densities']
  },
  {
    id: 'hierarchical',
    name: 'Hierarchical Clustering',
    category: 'Clustering',
    description: 'Creates hierarchy of clusters (dendrogram)',
    bestFor: ['Dendrogram visualization', 'Nested clusters', 'Small-medium data'],
    scalingRequired: true,
    speed: 2,
    accuracy: 3,
    interpretability: 5,
    hyperparameters: ['n_clusters', 'linkage', 'metric'],
    pros: ['No need to pre-specify K', 'Visualizable', 'Deterministic'],
    cons: ['Slow O(n³)', 'Memory intensive', 'Doesn\'t scale']
  },
  {
    id: 'gmm',
    name: 'Gaussian Mixture Models (GMM)',
    category: 'Clustering',
    description: 'Probabilistic clustering with soft assignments',
    bestFor: ['Soft clustering', 'Overlapping clusters', 'Probabilistic assignments'],
    scalingRequired: true,
    speed: 3,
    accuracy: 4,
    interpretability: 3,
    hyperparameters: ['n_components', 'covariance_type', 'max_iter', 'n_init'],
    pros: ['Probabilistic', 'Flexible cluster shapes', 'Soft assignments'],
    cons: ['Sensitive to initialization', 'Can overfit', 'Need to specify K']
  },

  // Dimensionality Reduction
  {
    id: 'pca',
    name: 'PCA',
    category: 'Dimensionality Reduction',
    description: 'Principal Component Analysis for linear dimensionality reduction',
    bestFor: ['Feature reduction', 'Visualization', 'Noise reduction', 'Preprocessing'],
    scalingRequired: true,
    speed: 5,
    accuracy: 4,
    interpretability: 4,
    hyperparameters: ['n_components', 'whiten', 'svd_solver'],
    pros: ['Fast', 'Interpretable', 'Preserves variance', 'Reduces dimensions'],
    cons: ['Linear only', 'Assumes orthogonal components']
  },
  {
    id: 'lda',
    name: 'LDA',
    category: 'Dimensionality Reduction',
    description: 'Linear Discriminant Analysis for supervised reduction',
    bestFor: ['Classification preprocessing', 'Class separation', 'Supervised reduction'],
    scalingRequired: true,
    speed: 4,
    accuracy: 4,
    interpretability: 4,
    hyperparameters: ['solver', 'shrinkage', 'n_components'],
    pros: ['Maximizes class separability', 'Supervised', 'Interpretable'],
    cons: ['Requires labels', 'Max n_components = n_classes - 1', 'Linear']
  },
  {
    id: 'svd',
    name: 'SVD (Truncated)',
    category: 'Dimensionality Reduction',
    description: 'Singular Value Decomposition for matrix factorization',
    bestFor: ['Sparse matrices', 'Text data (LSA)', 'Recommender systems'],
    scalingRequired: true,
    speed: 4,
    accuracy: 4,
    interpretability: 3,
    hyperparameters: ['n_components', 'algorithm', 'n_iter'],
    pros: ['Works with sparse data', 'Like PCA but no centering', 'Fast'],
    cons: ['Less interpretable than PCA']
  },
  {
    id: 'tsne',
    name: 't-SNE',
    category: 'Dimensionality Reduction',
    description: 't-distributed Stochastic Neighbor Embedding for visualization',
    bestFor: ['2D/3D visualization', 'Exploring clusters', 'Non-linear reduction'],
    scalingRequired: true,
    speed: 1,
    accuracy: 5,
    interpretability: 2,
    hyperparameters: ['n_components', 'perplexity', 'learning_rate', 'n_iter'],
    pros: ['Excellent visualization', 'Preserves local structure', 'Reveals clusters'],
    cons: ['Very slow', 'Non-deterministic', 'Can\'t transform new data easily']
  },
  {
    id: 'umap',
    name: 'UMAP',
    category: 'Dimensionality Reduction',
    description: 'Uniform Manifold Approximation for fast non-linear reduction',
    bestFor: ['Visualization', 'Faster than t-SNE', 'Large datasets'],
    scalingRequired: true,
    speed: 3,
    accuracy: 5,
    interpretability: 2,
    hyperparameters: ['n_components', 'n_neighbors', 'min_dist', 'metric'],
    pros: ['Faster than t-SNE', 'Preserves global + local structure', 'Scalable'],
    cons: ['Complex hyperparameters', 'Stochastic']
  }
];

interface AlgorithmExplorerProps {
  onBack?: () => void;
}

const AlgorithmExplorer: React.FC<AlgorithmExplorerProps> = ({ onBack }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(null);

  const categories = ['All', 'Classification', 'Regression', 'Clustering', 'Dimensionality Reduction'];

  const filteredAlgorithms = algorithms.filter(algo => {
    const matchesSearch = algo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         algo.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || algo.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const getCategoryStats = () => {
    return categories.slice(1).map(cat => ({
      name: cat,
      count: algorithms.filter(a => a.category === cat).length
    }));
  };

  const renderStars = (count: number) => {
    return (
      <div className="flex gap-0.5">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className={`w-2 h-2 rounded-full ${
              i < count ? 'bg-yellow-400' : 'bg-gray-300'
            }`}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Back Button */}
        {onBack && (
          <button
            onClick={onBack}
            className="mb-6 flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition border border-cyan-500/30"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Generator
          </button>
        )}

        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-16 h-16 text-cyan-400" />
          </div>
          <h1 className="text-6xl font-black text-white mb-4 uppercase tracking-tight">
            [<span className="text-cyan-400 glow-text">Algorithm</span>] Explorer
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Explore all 27+ machine learning algorithms with detailed information, use cases, and performance metrics
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
          <div className="bg-gray-900 rounded-2xl shadow-xl p-8 text-center border-l-4 border-cyan-500 hover:scale-105 transition-transform">
            <div className="text-4xl font-black text-cyan-400 mb-2">{algorithms.length}</div>
            <div className="text-xs text-gray-400 uppercase tracking-[0.2em] font-bold">Total Algorithms</div>
          </div>
          {getCategoryStats().map((stat, index) => (
            <div key={stat.name} className={`bg-gray-900 rounded-2xl shadow-xl p-8 text-center border-l-4 hover:scale-105 transition-transform ${
              index === 0 ? 'border-green-500' :
              index === 1 ? 'border-purple-500' :
              index === 2 ? 'border-orange-500' : 'border-pink-500'
            }`}>
              <div className={`text-4xl font-black mb-2 ${
                index === 0 ? 'text-green-400' :
                index === 1 ? 'text-purple-400' :
                index === 2 ? 'text-orange-400' : 'text-pink-400'
              }`}>{stat.count}</div>
              <div className="text-xs text-gray-400 uppercase tracking-[0.2em] font-bold">{stat.name}</div>
            </div>
          ))}
        </div>

        {/* Search and Filter */}
        <div className="bg-gray-900 rounded-2xl shadow-xl p-6 mb-8 border border-cyan-500/30">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 w-5 h-5" />
              <input
                type="text"
                placeholder="Search algorithms..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-black border-b-2 border-cyan-500/30 focus:border-cyan-500 focus:outline-none text-white rounded-t-lg"
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              {categories.map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-4 py-3 rounded-full font-bold transition uppercase tracking-wide text-sm ${
                    selectedCategory === cat
                      ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/30'
                      : 'bg-black/50 text-gray-400 hover:text-white border border-gray-700'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Algorithm Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {filteredAlgorithms.map((algo) => (
            <div
              key={algo.id}
              onClick={() => setSelectedAlgorithm(algo)}
              className="bg-gray-900 rounded-xl shadow-lg p-6 hover:shadow-2xl transition cursor-pointer border-2 border-gray-800 hover:border-cyan-500"
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="text-xl font-bold text-white">{algo.name}</h3>
                <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${
                  algo.category === 'Classification' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                  algo.category === 'Regression' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' :
                  algo.category === 'Clustering' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                  'bg-pink-500/20 text-pink-400 border border-pink-500/30'
                }`}>
                  {algo.category}
                </span>
              </div>

              <p className="text-gray-400 text-sm mb-4 line-clamp-2">{algo.description}</p>

              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 flex items-center gap-1 uppercase tracking-wide">
                    <Zap className="w-3 h-3" /> Speed
                  </span>
                  {renderStars(algo.speed)}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 flex items-center gap-1 uppercase tracking-wide">
                    <Target className="w-3 h-3" /> Accuracy
                  </span>
                  {renderStars(algo.accuracy)}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 flex items-center gap-1 uppercase tracking-wide">
                    <Eye className="w-3 h-3" /> Interpretability
                  </span>
                  {renderStars(algo.interpretability)}
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs">
                {algo.scalingRequired ? (
                  <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded border border-yellow-500/30">⚖️ Scaling Required</span>
                ) : (
                  <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded border border-green-500/30">✓ No Scaling</span>
                )}
              </div>
            </div>
          ))}
        </div>

        {filteredAlgorithms.length === 0 && (
          <div className="text-center py-12">
            <Brain className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <p className="text-xl text-gray-400">No algorithms found matching your search</p>
          </div>
        )}

        {/* Algorithm Detail Modal */}
        {selectedAlgorithm && (
          <div
            className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedAlgorithm(null)}
          >
            <div
              className="bg-gray-900 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-cyan-500/30"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 bg-gradient-to-r from-cyan-600 to-blue-600 text-white p-6 rounded-t-2xl">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-3xl font-black mb-2 uppercase tracking-wide">{selectedAlgorithm.name}</h2>
                    <span className="px-3 py-1 bg-white bg-opacity-20 rounded-full text-sm font-bold uppercase tracking-wide">
                      {selectedAlgorithm.category}
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedAlgorithm(null)}
                    className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-6">
                <div>
                  <h3 className="text-xl font-bold mb-2 text-cyan-400 uppercase tracking-wide">Description</h3>
                  <p className="text-gray-300">{selectedAlgorithm.description}</p>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-blue-500/20 rounded-lg p-4 border border-blue-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-5 h-5 text-blue-400" />
                      <span className="font-semibold text-white uppercase tracking-wide text-sm">Speed</span>
                    </div>
                    {renderStars(selectedAlgorithm.speed)}
                  </div>
                  <div className="bg-green-500/20 rounded-lg p-4 border border-green-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-5 h-5 text-green-400" />
                      <span className="font-semibold text-white uppercase tracking-wide text-sm">Accuracy</span>
                    </div>
                    {renderStars(selectedAlgorithm.accuracy)}
                  </div>
                  <div className="bg-purple-500/20 rounded-lg p-4 border border-purple-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="w-5 h-5 text-purple-400" />
                      <span className="font-semibold text-white uppercase tracking-wide text-sm">Interpretability</span>
                    </div>
                    {renderStars(selectedAlgorithm.interpretability)}
                  </div>
                </div>

                <div>
                  <h3 className="text-xl font-bold mb-3 text-cyan-400 uppercase tracking-wide">Best For</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedAlgorithm.bestFor.map((use, idx) => (
                      <span key={idx} className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm border border-cyan-500/30">
                        {use}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-xl font-bold mb-3 text-cyan-400 uppercase tracking-wide">Key Hyperparameters</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedAlgorithm.hyperparameters.map((param, idx) => (
                      <span key={idx} className="px-3 py-1 bg-black/50 text-gray-300 rounded-lg text-sm font-mono border border-gray-700">
                        {param}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-xl font-bold mb-3 text-green-400 uppercase tracking-wide">✓ Pros</h3>
                    <ul className="space-y-2">
                      {selectedAlgorithm.pros.map((pro, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-300">
                          <span className="text-green-400 mt-1">•</span>
                          <span>{pro}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-3 text-red-400 uppercase tracking-wide">✗ Cons</h3>
                    <ul className="space-y-2">
                      {selectedAlgorithm.cons.map((con, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-300">
                          <span className="text-red-400 mt-1">•</span>
                          <span>{con}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className={`p-4 rounded-lg ${
                  selectedAlgorithm.scalingRequired ? 'bg-yellow-500/20 border-2 border-yellow-500/30' : 'bg-green-500/20 border-2 border-green-500/30'
                }`}>
                  <div className="flex items-center gap-2">
                    <Layers className={`w-5 h-5 ${selectedAlgorithm.scalingRequired ? 'text-yellow-400' : 'text-green-400'}`} />
                    <span className="font-semibold text-white uppercase tracking-wide">
                      {selectedAlgorithm.scalingRequired
                        ? '⚖️ Feature Scaling Required'
                        : '✓ No Feature Scaling Needed'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 mt-2">
                    {selectedAlgorithm.scalingRequired
                      ? 'This algorithm is distance-based and requires features to be scaled (StandardScaler, MinMaxScaler, etc.)'
                      : 'Tree-based algorithm that doesn\'t require feature scaling'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AlgorithmExplorer;
