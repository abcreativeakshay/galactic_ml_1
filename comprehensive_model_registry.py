"""
Comprehensive Model Registry for All ML Algorithms
Covers 27+ algorithms with proper hyperparameter spaces
"""

import numpy as np
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class ComprehensiveModelRegistry:
    """
    Registry for all ML algorithms with their configurations
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_classification_models(self):
        """Get all classification models with hyperparameter spaces"""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'scaling_required': True,
                'description': 'Linear model for binary/multiclass classification'
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'scaling_required': False,
                'description': 'Ensemble of decision trees'
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None]
                },
                'scaling_required': False,
                'description': 'Tree-based classifier'
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                'scaling_required': True,
                'description': 'Instance-based learning algorithm'
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                },
                'scaling_required': False,
                'description': 'Probabilistic classifier based on Bayes theorem'
            },
            'SVM': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'scaling_required': True,
                'description': 'Support Vector Machine classifier'
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                'scaling_required': False,
                'description': 'Gradient boosting ensemble'
            },
            'MLP Neural Network': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'solver': ['adam', 'sgd']
                },
                'scaling_required': True,
                'description': 'Multi-layer Perceptron neural network'
            }
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.5]
                },
                'scaling_required': False,
                'description': 'Extreme Gradient Boosting'
            }

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [15, 31, 63, 127],
                    'min_child_samples': [5, 10, 20]
                },
                'scaling_required': False,
                'description': 'Light Gradient Boosting Machine'
            }

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = {
                'model': CatBoostClassifier(random_state=self.random_state, verbose=0),
                'params': {
                    'iterations': [50, 100, 200],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'l2_leaf_reg': [1, 3, 5, 7]
                },
                'scaling_required': False,
                'description': 'Categorical Boosting'
            }

        return models

    def get_regression_models(self):
        """Get all regression models with hyperparameter spaces"""
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                },
                'scaling_required': True,
                'description': 'Ordinary Least Squares regression'
            },
            'Ridge Regression': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                'scaling_required': True,
                'description': 'L2 regularized linear regression'
            },
            'Lasso Regression': {
                'model': Lasso(random_state=self.random_state, max_iter=10000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'selection': ['cyclic', 'random']
                },
                'scaling_required': True,
                'description': 'L1 regularized linear regression'
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=self.random_state, max_iter=10000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'selection': ['cyclic', 'random']
                },
                'scaling_required': True,
                'description': 'L1 + L2 regularized linear regression'
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'scaling_required': False,
                'description': 'Ensemble of decision trees for regression'
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                },
                'scaling_required': False,
                'description': 'Tree-based regressor'
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'scaling_required': True,
                'description': 'Instance-based learning for regression'
            },
            'SVM': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.5]
                },
                'scaling_required': True,
                'description': 'Support Vector Regression'
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                'scaling_required': False,
                'description': 'Gradient boosting for regression'
            },
            'MLP Neural Network': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'solver': ['adam', 'lbfgs']
                },
                'scaling_required': True,
                'description': 'Multi-layer Perceptron for regression'
            }
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': XGBRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                'scaling_required': False,
                'description': 'Extreme Gradient Boosting for regression'
            }

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': LGBMRegressor(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [15, 31, 63, 127]
                },
                'scaling_required': False,
                'description': 'Light GBM for regression'
            }

        return models

    def get_clustering_models(self):
        """Get all clustering models with hyperparameter spaces"""
        models = {
            'K-Means': {
                'model': KMeans(random_state=self.random_state),
                'params': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'init': ['k-means++', 'random'],
                    'max_iter': [100, 300, 500],
                    'n_init': [10, 20]
                },
                'scaling_required': True,
                'description': 'K-means clustering algorithm'
            },
            'DBSCAN': {
                'model': DBSCAN(),
                'params': {
                    'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
                    'min_samples': [3, 5, 10, 15],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                'scaling_required': True,
                'description': 'Density-based clustering'
            },
            'Hierarchical': {
                'model': AgglomerativeClustering(),
                'params': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                'scaling_required': True,
                'description': 'Hierarchical clustering'
            },
            'Gaussian Mixture': {
                'model': GaussianMixture(random_state=self.random_state),
                'params': {
                    'n_components': [2, 3, 4, 5, 6, 7, 8],
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'max_iter': [100, 200],
                    'n_init': [1, 5, 10]
                },
                'scaling_required': True,
                'description': 'Gaussian Mixture Model clustering'
            }
        }

        return models

    def get_dimensionality_reduction_models(self):
        """Get all dimensionality reduction models"""
        models = {
            'PCA': {
                'model': PCA(random_state=self.random_state),
                'params': {
                    'n_components': [2, 3, 5, 10, None],
                    'whiten': [True, False],
                    'svd_solver': ['auto', 'full', 'arpack', 'randomized']
                },
                'scaling_required': True,
                'description': 'Principal Component Analysis'
            },
            'LDA': {
                'model': LinearDiscriminantAnalysis(),
                'params': {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
                },
                'scaling_required': True,
                'description': 'Linear Discriminant Analysis'
            },
            'SVD': {
                'model': TruncatedSVD(random_state=self.random_state),
                'params': {
                    'n_components': [2, 5, 10, 20, 50],
                    'algorithm': ['arpack', 'randomized'],
                    'n_iter': [5, 10, 20]
                },
                'scaling_required': True,
                'description': 'Truncated Singular Value Decomposition'
            },
            't-SNE': {
                'model': TSNE(random_state=self.random_state),
                'params': {
                    'n_components': [2, 3],
                    'perplexity': [5, 30, 50, 100],
                    'learning_rate': [10, 50, 100, 200, 500],
                    'n_iter': [250, 500, 1000],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                'scaling_required': True,
                'description': 't-distributed Stochastic Neighbor Embedding'
            }
        }

        # Add UMAP if available
        if UMAP_AVAILABLE:
            models['UMAP'] = {
                'model': umap.UMAP(random_state=self.random_state),
                'params': {
                    'n_components': [2, 3],
                    'n_neighbors': [5, 15, 30, 50],
                    'min_dist': [0.0, 0.1, 0.25, 0.5],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                'scaling_required': True,
                'description': 'Uniform Manifold Approximation and Projection'
            }

        return models

    def get_models_for_problem_type(self, problem_type):
        """Get appropriate models for the given problem type"""
        if problem_type in ['binary_classification', 'multiclass_classification']:
            return self.get_classification_models()
        elif problem_type == 'regression':
            return self.get_regression_models()
        elif problem_type == 'clustering':
            return self.get_clustering_models()
        elif problem_type == 'dimensionality_reduction':
            return self.get_dimensionality_reduction_models()
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def print_model_summary(self):
        """Print summary of all available models"""
        print("="*80)
        print("📚 COMPREHENSIVE MODEL REGISTRY SUMMARY")
        print("="*80)

        categories = [
            ('Classification', self.get_classification_models()),
            ('Regression', self.get_regression_models()),
            ('Clustering', self.get_clustering_models()),
            ('Dimensionality Reduction', self.get_dimensionality_reduction_models())
        ]

        total_models = 0
        for category, models in categories:
            print(f"\n🔹 {category} ({len(models)} models):")
            for name, config in models.items():
                n_params = len(config['params'])
                scaling = "⚖️  Scaling Required" if config.get('scaling_required', False) else "🔓 No Scaling"
                print(f"   • {name}: {n_params} hyperparameters | {scaling}")
                print(f"     └─ {config['description']}")
            total_models += len(models)

        print(f"\n{'='*80}")
        print(f"✅ Total Models Available: {total_models}")
        print(f"{'='*80}")


if __name__ == "__main__":
    registry = ComprehensiveModelRegistry()
    registry.print_model_summary()
