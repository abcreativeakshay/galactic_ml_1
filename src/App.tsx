import React, { useState } from 'react';
import { Upload, Download, Brain, FileJson, FileCode, Loader2, CheckCircle, AlertCircle, BookOpen, Home } from 'lucide-react';
import Papa from 'papaparse';
import { analyzeDataset } from './services/geminiService';
import { generateNotebook, convertNotebookToPython, type NotebookConfig } from './services/notebookGenerator';
import AlgorithmExplorer from './components/AlgorithmExplorer';

interface DatasetInfo {
  name: string;
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  sample: any[];
}

type GenerationStep = 'idle' | 'uploading' | 'analyzing' | 'configuring' | 'generating' | 'complete' | 'error';
type View = 'home' | 'explorer';

function App() {
  const [currentView, setCurrentView] = useState<View>('home');
  const [step, setStep] = useState<GenerationStep>('idle');
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [problemType, setProblemType] = useState<string>('classification');
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([]);
  const [generatedNotebook, setGeneratedNotebook] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [progress, setProgress] = useState<string>('');

  const mlModels: Record<string, string[]> = {
    'classification': [
      'Logistic Regression', 'Random Forest', 'Decision Tree', 'KNN', 'Naive Bayes',
      'SVM', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'MLP Neural Network'
    ],
    'regression': [
      'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet',
      'Random Forest', 'Decision Tree', 'KNN', 'SVR', 'XGBoost', 'MLP Neural Network'
    ],
    'clustering': [
      'K-Means', 'DBSCAN', 'Hierarchical Clustering', 'Gaussian Mixture Models'
    ]
  };

  const availableModels = mlModels[problemType] || [];

  const techniqueCategories: Record<string, Record<string, string[]>> = {
    'Dimensionality Reduction': {
      'Linear Methods': [
        'Principal Component Analysis (PCA)', 'Linear Discriminant Analysis (LDA)',
        'Factor Analysis', 'Independent Component Analysis (ICA)',
        'Non-negative Matrix Factorization (NMF)', 'Truncated SVD', 'Kernel PCA'
      ],
      'Non-linear Methods': [
        't-SNE', 'UMAP', 'Isomap', 'Locally Linear Embedding (LLE)',
        'Multidimensional Scaling (MDS)', 'Autoencoders', 'Self-Organizing Maps (SOM)'
      ]
    },
    'Feature Selection': {
      'Filter Methods': [
        'Variance Threshold', 'Correlation-based Feature Selection', 'Chi-Square Test',
        'ANOVA F-test', 'Mutual Information', 'Fisher Score'
      ],
      'Wrapper Methods': [
        'Recursive Feature Elimination (RFE)', 'Forward Selection', 'Backward Elimination',
        'Bidirectional Elimination', 'Exhaustive Feature Selection'
      ],
      'Embedded Methods': [
        'L1 Regularization (Lasso)', 'Ridge Regression', 'Elastic Net',
        'Decision Tree Feature Importance', 'Random Forest Feature Importance'
      ]
    },
    'Data Preprocessing': {
      'Scaling & Normalization': [
        'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
        'Power Transformers', 'Quantile Transformer', 'Log Transformation'
      ],
      'Encoding Categorical': [
        'One-Hot Encoding', 'Label Encoding', 'Ordinal Encoding', 'Target Encoding',
        'Frequency Encoding', 'Binary Encoding', 'Hashing Trick'
      ],
      'Missing Values': [
        'Mean/Median/Mode Imputation', 'K-Nearest Neighbors Imputation',
        'Iterative Imputer (MICE)', 'MissForest', 'Constant Value Imputation',
        'Predictive Imputation'
      ]
    },
    'Imbalanced Data': {
      'Oversampling': [
        'Random Oversampling', 'SMOTE', 'ADASYN', 'Borderline-SMOTE',
        'SVM-SMOTE', 'K-Means SMOTE'
      ],
      'Undersampling': [
        'Random Undersampling', 'Tomek Links', 'Cluster Centroids',
        'NearMiss', 'Edited Nearest Neighbors', 'One-Sided Selection'
      ],
      'Hybrid Methods': [
        'SMOTE + Tomek Links', 'SMOTE + Edited Nearest Neighbors'
      ]
    },
    'Data Augmentation': {
      'Image': [
        'Rotation', 'Flipping', 'Cropping', 'Scaling', 'Shearing', 'Zooming',
        'Color Jittering', 'Brightness/Contrast Adjustment', 'Gaussian Noise',
        'Cutout', 'MixUp', 'CutMix'
      ],
      'Text': [
        'Synonym Replacement', 'Random Insertion', 'Random Swap', 'Random Deletion',
        'Back Translation', 'TF-IDF Replacement', 'Contextual Word Embeddings Replacement'
      ],
      'Audio': [
        'Time Stretching', 'Pitch Shifting', 'Adding Noise', 'Time Shifting',
        'Speed Change', 'Volume Change'
      ]
    },
    'Regularization': {
      'Traditional ML': [
        'L1 Regularization (Lasso)', 'L2 Regularization (Ridge)', 'Elastic Net',
        'Early Stopping', 'Dropout'
      ],
      'Deep Learning': [
        'Dropout', 'DropConnect', 'Batch Normalization', 'Layer Normalization',
        'Weight Normalization', 'Group Normalization', 'Instance Normalization',
        'Weight Decay', 'Label Smoothing', 'Stochastic Depth', 'Shake-Shake Regularization',
        'Cutout', 'MixUp', 'Manifold MixUp'
      ]
    },
    'Ensemble Methods': {
      'Averaging-based': [
        'Bagging', 'Random Forest', 'Extra Trees', 'Pasting',
        'Random Subspaces', 'Random Patches'
      ],
      'Boosting': [
        'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost',
        'Stochastic Gradient Boosting', 'LPBoost', 'BrownBoost'
      ],
      'Stacking & Blending': [
        'Stacking', 'Blending', 'Voting Classifiers', 'Super Learner'
      ]
    },
    'Optimization': {
      'Gradient-based': [
        'Stochastic Gradient Descent (SGD)', 'Momentum', 'Nesterov Accelerated Gradient',
        'AdaGrad', 'RMSProp', 'Adam', 'AdamW', 'AdaDelta', 'Nadam', 'AMSGrad'
      ],
      'Learning Rate': [
        'Step Decay', 'Exponential Decay', 'Cosine Annealing',
        'Cosine Annealing with Warm Restarts', 'Cyclical Learning Rates',
        'One-Cycle Learning Rate', 'Polynomial Decay', 'Time-based Decay'
      ],
      'Hyperparameter': [
        'Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband',
        'BOHB', 'Genetic Algorithms', 'Particle Swarm Optimization',
        'Tree-structured Parzen Estimators'
      ]
    },
    'Loss Functions': {
      'Classification': [
        'Binary Cross-Entropy', 'Categorical Cross-Entropy', 'Sparse Categorical Cross-Entropy',
        'Focal Loss', 'Hinge Loss', 'Squared Hinge Loss', 'Kullback-Leibler Divergence'
      ],
      'Regression': [
        'Mean Squared Error', 'Mean Absolute Error', 'Huber Loss',
        'Log-Cosh Loss', 'Quantile Loss', 'Mean Absolute Percentage Error'
      ],
      'Specialized': [
        'Triplet Loss', 'Contrastive Loss', 'Center Loss', 'ArcFace Loss',
        'CosFace Loss', 'SphereFace Loss', 'Dice Loss', 'IoU Loss', 'Focal Tversky Loss'
      ]
    },
    'Attention Mechanisms': {
      'All Types': [
        'Self-Attention', 'Multi-Head Attention', 'Scaled Dot-Product Attention',
        'Additive Attention', 'Local Attention', 'Global Attention',
        'Hierarchical Attention', 'Cross-Attention', 'Sparse Attention',
        'Linear Attention', 'Performer Attention'
      ]
    },
    'Transfer Learning': {
      'Fine-tuning': [
        'Feature Extraction', 'Full Fine-tuning', 'Gradual Unfreezing',
        'Discriminative Learning Rates', 'Layer-wise Learning Rate Decay'
      ],
      'Parameter-efficient': [
        'LoRA', 'Adapter Layers', 'Prefix Tuning', 'Prompt Tuning',
        'BitFit', 'Compacter'
      ],
      'Domain Adaptation': [
        'Domain Adversarial Training', 'Domain Separation Networks',
        'Correlation Alignment', 'Maximum Mean Discrepancy'
      ]
    },
    'Evaluation': {
      'Cross-Validation': [
        'K-Fold Cross-Validation', 'Stratified K-Fold', 'Leave-One-Out Cross-Validation',
        'Leave-P-Out Cross-Validation', 'Time Series Split', 'Group K-Fold',
        'Repeated K-Fold', 'Nested Cross-Validation'
      ],
      'Sampling': [
        'Bootstrap Sampling', 'Stratified Sampling', 'Cluster Sampling', 'Systematic Sampling'
      ],
      'Metrics': [
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC',
        'Mean Absolute Error', 'Mean Squared Error', 'R-squared', 'Adjusted R-squared',
        'Silhouette Score', 'Davies-Bouldin Index', 'Perplexity', 'BLEU Score', 'ROUGE Score'
      ]
    },
    'Training Techniques': {
      'All Methods': [
        'Curriculum Learning', 'Self-Training', 'Co-Training', 'Multi-Task Learning',
        'Meta-Learning', 'Knowledge Distillation', 'Pruning', 'Quantization',
        'Neural Architecture Search', 'Differentiable Architecture Search'
      ]
    },
    'Contrastive Learning': {
      'Methods': [
        'SimCLR', 'MoCo', 'BYOL', 'SwAV', 'Barlow Twins', 'VicReg', 'SupCon'
      ]
    }
  };

  const [techniqueCategory, setTechniqueCategory] = useState<string>('Dimensionality Reduction');
  const [techniqueSubcategory, setTechniqueSubcategory] = useState<string>('Linear Methods');

  const availableTechniqueCategories = Object.keys(techniqueCategories);
  const availableTechniqueSubcategories = Object.keys(techniqueCategories[techniqueCategory] || {});
  const availableTechniques = techniqueCategories[techniqueCategory]?.[techniqueSubcategory] || [];

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setStep('uploading');
    setError('');
    setProgress('Reading dataset...');

    try {
      const text = await file.text();

      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: async (results) => {
          const data = results.data;
          const columns = results.meta.fields || [];

          const dtypes: Record<string, string> = {};
          columns.forEach(col => {
            const firstValue = data[0]?.[col];
            dtypes[col] = typeof firstValue === 'number' ? 'numeric' : 'categorical';
          });

          const info: DatasetInfo = {
            name: file.name,
            shape: [data.length, columns.length],
            columns,
            dtypes,
            sample: data.slice(0, 5)
          };

          setDatasetInfo(info);
          setStep('analyzing');
          setProgress('Analyzing dataset with AI...');

          try {
            const analysis = await analyzeDataset(info.sample);
            setProblemType(analysis.problemType);

            if (analysis.suggestedTarget.length > 0) {
              setTargetColumn(analysis.suggestedTarget[0]);
            }
            setSelectedModels(analysis.recommendedModels.slice(0, 3));

            setStep('configuring');
            setProgress('');
          } catch (err) {
            console.error('Analysis error:', err);
            setStep('configuring');
          }
        },
        error: (err) => {
          setError(`Failed to parse CSV: ${err.message}`);
          setStep('error');
        }
      });
    } catch (err) {
      setError(`Failed to read file: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setStep('error');
    }
  };

  const handleModelToggle = (model: string) => {
    setSelectedModels(prev =>
      prev.includes(model)
        ? prev.filter(m => m !== model)
        : [...prev, model]
    );
  };

  const handleTechniqueToggle = (technique: string) => {
    setSelectedTechniques(prev =>
      prev.includes(technique)
        ? prev.filter(t => t !== technique)
        : [...prev, technique]
    );
  };

  const handleGenerate = async () => {
    if (!datasetInfo || !targetColumn || selectedModels.length === 0) {
      setError('Please configure all required fields');
      return;
    }

    setStep('generating');
    setError('');
    setProgress('Initializing notebook generation...');

    try {
      const config: NotebookConfig = {
        datasetName: datasetInfo.name,
        problemType,
        targetColumn,
        selectedModels,
        selectedTechniques,
        datasetInfo: {
          shape: datasetInfo.shape,
          columns: datasetInfo.columns,
          dtypes: datasetInfo.dtypes
        }
      };

      setProgress('Generating ML theory sections with AI...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProgress('Creating notebook cells...');
      const notebook = await generateNotebook(config);

      setProgress('Finalizing notebook...');
      setGeneratedNotebook(notebook);
      setStep('complete');
      setProgress('Notebook generated successfully!');
    } catch (err) {
      console.error('Generation error:', err);
      setError(`Failed to generate notebook: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setStep('error');
    }
  };

  const downloadNotebook = () => {
    if (!generatedNotebook) return;

    const blob = new Blob([JSON.stringify(generatedNotebook, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_pipeline_${Date.now()}.ipynb`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadPython = () => {
    if (!generatedNotebook) return;

    const pyContent = convertNotebookToPython(generatedNotebook);
    const blob = new Blob([pyContent], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_pipeline_${Date.now()}.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const reset = () => {
    setStep('idle');
    setDatasetInfo(null);
    setProblemType('classification');
    setTargetColumn('');
    setSelectedModels([]);
    setGeneratedNotebook(null);
    setError('');
    setProgress('');
  };

  if (currentView === 'explorer') {
    return <AlgorithmExplorer onBack={() => setCurrentView('home')} />;
  }

  return (
    <div className="min-h-screen bg-black">
      {/* Header Navigation */}
      <header className="bg-black border-b border-gray-800/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Brain className="w-10 h-10 text-cyan-400 animate-pulse-glow" />
              <div className="absolute inset-0 bg-cyan-400/20 blur-xl rounded-full"></div>
            </div>
            <span className="text-2xl font-black text-white tracking-tight">GALACTIC<span className="text-cyan-400">ML</span></span>
          </div>
          <nav className="hidden md:flex items-center gap-8">
            <button
              onClick={() => setCurrentView('home')}
              className={`${currentView === 'home' ? 'text-cyan-400' : 'text-gray-400 hover:text-white'} transition uppercase tracking-wider text-sm font-bold`}
            >
              Generator
            </button>
            <button
              onClick={() => setCurrentView('explorer')}
              className={`${currentView === 'explorer' ? 'text-cyan-400' : 'text-gray-400 hover:text-white'} transition uppercase tracking-wider text-sm font-bold`}
            >
              Algorithms
            </button>
            <button className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white px-8 py-3 rounded-full font-black transition uppercase tracking-wider text-xs shadow-lg shadow-cyan-500/50">
              Get Started
            </button>
          </nav>
        </div>
      </header>

      <div>
        {step === 'idle' && (
          <div className="relative overflow-hidden min-h-screen flex items-center">
            {/* Animated Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black"></div>
            <div className="absolute inset-0 cyber-grid opacity-30"></div>
            <div className="absolute inset-0 diagonal-lines opacity-20"></div>

            {/* Animated Gradient Orbs */}
            <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-gradient-to-br from-cyan-500/20 to-transparent rounded-full blur-3xl animate-pulse"></div>
            <div className="absolute bottom-0 right-0 w-[800px] h-[800px] bg-gradient-to-tl from-blue-500/20 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>

            {/* Floating Particles */}
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="particle"
                style={{
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 20}s`,
                  animationDuration: `${15 + Math.random() * 10}s`
                }}
              />
            ))}

            <div className="relative w-full max-w-7xl mx-auto px-6 py-24">
              <div className="grid lg:grid-cols-2 gap-20 items-center">
                {/* Left Content */}
                <div className="text-left space-y-10">
                  {/* Badge */}
                  <div className="inline-block animate-fade-in-up">
                    <div className="relative group">
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full blur-lg opacity-50 group-hover:opacity-75 transition"></div>
                      <div className="relative text-xs font-black text-cyan-400 uppercase tracking-[0.3em] px-6 py-3 border-2 border-cyan-500/50 rounded-full bg-black/50 backdrop-blur-xl hover:border-cyan-400 transition">
                        🚀 AI-Powered ML Platform
                      </div>
                    </div>
                  </div>

                  {/* Main Heading */}
                  <div className="space-y-6 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
                    <div className="relative">
                      <h1 className="text-8xl font-black text-white leading-none tracking-tighter">
                        <span className="inline-block">[</span>
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-cyan-400 animate-gradient glow-text">
                          GALACTIC
                        </span>
                        <span className="inline-block">]</span>
                      </h1>
                      <div className="absolute -top-4 -left-4 w-24 h-24 bg-cyan-500/20 rounded-full blur-2xl"></div>
                    </div>

                    <h2 className="text-5xl font-black text-white leading-tight tracking-tight">
                      Build ML Notebooks<br />
                      <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 animate-gradient">
                        In The Age Of AI
                      </span>
                    </h2>
                  </div>

                  {/* Description */}
                  <div className="space-y-6 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
                    <div className="relative">
                      <div className="absolute -left-4 top-0 w-1 h-full bg-gradient-to-b from-cyan-500 to-blue-500 rounded-full"></div>
                      <p className="text-2xl text-cyan-400 font-black uppercase tracking-wide mb-4">
                        Where ML Gets Real.
                      </p>
                      <p className="text-lg text-gray-400 leading-relaxed max-w-xl">
                        Upload your dataset and get production-ready Jupyter notebooks with AI-generated theory, hyperparameter tuning, and complete ML pipelines.
                      </p>
                    </div>
                  </div>

                  {/* Feature Pills */}
                  <div className="flex flex-wrap gap-3 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
                    {[
                      { icon: '🤖', text: '27+ Algorithms' },
                      { icon: '⚡', text: 'AI-Powered' },
                      { icon: '🚀', text: 'Production Ready' },
                      { icon: '🎯', text: 'Auto-Tuned' }
                    ].map((feature, idx) => (
                      <div key={idx} className="group relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-full blur-md group-hover:blur-lg transition"></div>
                        <div className="relative flex items-center gap-2 px-5 py-2.5 bg-gray-900/80 backdrop-blur-xl border border-cyan-500/30 rounded-full hover:border-cyan-400 transition">
                          <span className="text-lg">{feature.icon}</span>
                          <span className="text-sm font-bold text-gray-300 group-hover:text-white transition">{feature.text}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* CTA Section */}
                  <div className="flex items-center gap-4 pt-4 animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
                    <div className="flex items-center gap-3">
                      <div className="flex -space-x-2">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 border-2 border-black"></div>
                        ))}
                      </div>
                      <div className="text-sm">
                        <div className="font-bold text-white">10,000+ Notebooks</div>
                        <div className="text-gray-500">Generated this month</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Content - Upload Card */}
                <div className="relative animate-scale-in" style={{ animationDelay: '0.2s' }}>
                  {/* Glow Effects */}
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/30 to-blue-500/30 rounded-3xl blur-3xl"></div>
                  <div className="absolute inset-0 bg-gradient-to-tl from-purple-500/20 to-transparent rounded-3xl blur-2xl"></div>

                  {/* Card */}
                  <div className="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-3xl shadow-2xl p-12 border-2 border-cyan-500/30 backdrop-blur-xl hover:border-cyan-400 transition-all duration-500 group">
                    {/* Corner Accents */}
                    <div className="absolute top-0 left-0 w-20 h-20 border-t-4 border-l-4 border-cyan-500 rounded-tl-3xl"></div>
                    <div className="absolute bottom-0 right-0 w-20 h-20 border-b-4 border-r-4 border-blue-500 rounded-br-3xl"></div>

                    {/* Header */}
                    <div className="text-center mb-10">
                      <div className="inline-block mb-6 relative">
                        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-2xl blur-xl animate-pulse"></div>
                        <div className="relative w-24 h-24 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-2xl flex items-center justify-center mx-auto rotate-6 group-hover:rotate-12 transition-transform duration-500 shadow-2xl">
                          <Upload className="w-12 h-12 text-white" />
                        </div>
                      </div>
                      <h3 className="text-4xl font-black text-white mb-4 uppercase tracking-tight">
                        Start Building
                      </h3>
                      <p className="text-gray-400 text-base font-medium">
                        Upload your CSV dataset to begin your ML journey
                      </p>
                    </div>

                    {/* Upload Button */}
                    <div className="text-center mb-10">
                      <label className="inline-block group/btn">
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                        <div className="relative">
                          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full blur-xl group-hover/btn:blur-2xl transition-all"></div>
                          <span className="relative bg-gradient-to-r from-cyan-500 to-blue-500 text-white px-12 py-6 rounded-full text-xl font-black hover:from-cyan-400 hover:to-blue-400 transition-all cursor-pointer inline-flex items-center gap-4 uppercase tracking-wider shadow-2xl hover:shadow-cyan-500/50 hover:scale-105 transform">
                            <Upload className="w-7 h-7" />
                            Choose CSV File
                          </span>
                        </div>
                      </label>
                      <p className="text-gray-500 text-xs mt-4">Drag & drop also supported</p>
                    </div>

                    {/* Features Grid */}
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"></div>
                        <span className="font-black text-white text-xs uppercase tracking-widest">Included</span>
                        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"></div>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        {[
                          { icon: '🧠', text: 'AI Theory' },
                          { icon: '⚙️', text: 'Auto-Tune' },
                          { icon: '📊', text: 'Visualizations' },
                          { icon: '🎯', text: 'Evaluation' },
                          { icon: '🔄', text: 'Preprocessing' },
                          { icon: '🚀', text: 'Deployment' }
                        ].map((feature, idx) => (
                          <div key={idx} className="flex items-center gap-3 p-3 bg-black/30 rounded-xl border border-cyan-500/20 hover:border-cyan-400/50 transition group/item">
                            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-lg flex items-center justify-center text-xl group-hover/item:scale-110 transition">
                              {feature.icon}
                            </div>
                            <span className="font-bold text-gray-300 text-sm group-hover/item:text-white transition">{feature.text}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Bottom Stats */}
                    <div className="mt-8 pt-6 border-t border-cyan-500/20">
                      <div className="flex items-center justify-center gap-6 text-center">
                        <div>
                          <div className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">27+</div>
                          <div className="text-xs text-gray-500 uppercase tracking-wider">Algorithms</div>
                        </div>
                        <div className="w-px h-10 bg-cyan-500/20"></div>
                        <div>
                          <div className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">100%</div>
                          <div className="text-xs text-gray-500 uppercase tracking-wider">Automated</div>
                        </div>
                        <div className="w-px h-10 bg-cyan-500/20"></div>
                        <div>
                          <div className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">5min</div>
                          <div className="text-xs text-gray-500 uppercase tracking-wider">Generation</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {(step === 'uploading' || step === 'analyzing') && (
          <div className="max-w-4xl mx-auto px-6 py-20">
            <div className="bg-gray-900 rounded-2xl shadow-2xl p-12 text-center border border-cyan-500/30">
              <Loader2 className="w-20 h-20 text-cyan-500 mx-auto mb-6 animate-spin" />
              <h2 className="text-3xl font-bold text-white mb-4">{progress}</h2>
              <p className="text-gray-400">Please wait while we process your dataset...</p>
              <div className="mt-8 max-w-md mx-auto">
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse" style={{ width: '60%' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 'configuring' && datasetInfo && (
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="bg-gray-900 rounded-3xl shadow-2xl p-10 border-2 border-cyan-500/30 backdrop-blur-xl">
              <div className="text-center mb-12">
                <div className="inline-block mb-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-2xl flex items-center justify-center mx-auto shadow-lg shadow-cyan-500/50">
                    <Brain className="w-8 h-8 text-white" />
                  </div>
                </div>
                <h2 className="text-5xl font-black text-white mb-4 uppercase tracking-tight">Configure Pipeline</h2>
                <div className="h-1 w-32 bg-gradient-to-r from-cyan-500 to-blue-500 mx-auto rounded-full"></div>
              </div>

              <div className="mb-10 p-8 bg-black/50 rounded-2xl border-2 border-cyan-500/20 backdrop-blur-xl">
                <h3 className="font-black text-xl mb-6 text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                  Dataset Information
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500 uppercase text-xs tracking-wider">Name:</span>
                    <span className="ml-2 font-medium text-white">{datasetInfo.name}</span>
                  </div>
                  <div>
                    <span className="text-gray-500 uppercase text-xs tracking-wider">Shape:</span>
                    <span className="ml-2 font-medium text-white">{datasetInfo.shape[0]} rows × {datasetInfo.shape[1]} columns</span>
                  </div>
                </div>
              </div>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div>
                <label className="block text-sm font-bold text-cyan-400 mb-3 uppercase tracking-wide">
                  Problem Type
                </label>
                <select
                  value={problemType}
                  onChange={(e) => {
                    setProblemType(e.target.value);
                    setSelectedModels([]);
                  }}
                  className="w-full px-4 py-3 bg-black border-b-2 border-cyan-500/30 focus:border-cyan-500 focus:outline-none text-white rounded-t-lg"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                  <option value="clustering">Clustering</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-bold text-cyan-400 mb-3 uppercase tracking-wide">
                  Target Column
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-3 bg-black border-b-2 border-cyan-500/30 focus:border-cyan-500 focus:outline-none text-white rounded-t-lg"
                >
                  <option value="">Select target column...</option>
                  {datasetInfo.columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="mb-10 p-8 bg-gradient-to-br from-cyan-900/30 to-blue-900/30 rounded-2xl border-2 border-cyan-500/30 backdrop-blur-xl">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-cyan-500/20 rounded-xl flex items-center justify-center border border-cyan-500/30">
                    <Brain className="w-6 h-6 text-cyan-400" />
                  </div>
                  <h3 className="text-xl font-black text-white uppercase tracking-wider">Step 1: Select ML Models</h3>
                </div>
                <span className="text-xs text-gray-400 bg-black/50 px-3 py-1 rounded-full border border-cyan-500/30">
                  {availableModels.length} available • {selectedModels.length} selected
                </span>
              </div>
              <p className="text-sm text-gray-400 mb-4">Select 2-5 ML algorithms to train and compare</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-96 overflow-y-auto p-1">
                {availableModels.map(model => (
                  <button
                    key={model}
                    onClick={() => handleModelToggle(model)}
                    className={`px-4 py-3 rounded-lg border-2 transition text-sm font-semibold ${
                      selectedModels.includes(model)
                        ? 'bg-cyan-500 text-white border-cyan-500 shadow-lg shadow-cyan-500/50'
                        : 'bg-black/50 text-gray-300 border-gray-700 hover:border-cyan-500/50 hover:text-white'
                    }`}
                  >
                    {model}
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-10 p-8 bg-gradient-to-br from-blue-900/30 to-cyan-900/30 rounded-2xl border-2 border-blue-500/30 backdrop-blur-xl">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center border border-blue-500/30">
                    <FileCode className="w-6 h-6 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-black text-white uppercase tracking-wider">Step 2: Select Techniques</h3>
                </div>
                <span className="text-xs text-gray-400 bg-black/50 px-3 py-1 rounded-full border border-blue-500/30">
                  {availableTechniques.length} available • {selectedTechniques.length} selected
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-xs font-bold text-blue-400 mb-2 uppercase tracking-wider">
                    Technique Category
                  </label>
                  <select
                    value={techniqueCategory}
                    onChange={(e) => {
                      setTechniqueCategory(e.target.value);
                      const newSubcats = Object.keys(techniqueCategories[e.target.value] || {});
                      setTechniqueSubcategory(newSubcats[0] || '');
                    }}
                    className="w-full px-4 py-3 bg-black border-b-2 border-blue-500/30 focus:border-blue-500 focus:outline-none text-white rounded-t-lg"
                  >
                    {availableTechniqueCategories.map(cat => (
                      <option key={cat} value={cat}>{cat}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-bold text-blue-400 mb-2 uppercase tracking-wider">
                    Subcategory
                  </label>
                  <select
                    value={techniqueSubcategory}
                    onChange={(e) => setTechniqueSubcategory(e.target.value)}
                    className="w-full px-4 py-3 bg-black border-b-2 border-blue-500/30 focus:border-blue-500 focus:outline-none text-white rounded-t-lg"
                  >
                    {availableTechniqueSubcategories.map(subcat => (
                      <option key={subcat} value={subcat}>{subcat}</option>
                    ))}
                  </select>
                </div>
              </div>

              <p className="text-sm text-gray-400 mb-4">Select techniques to apply to your ML models</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-80 overflow-y-auto p-1">
                {availableTechniques.map(technique => (
                  <button
                    key={technique}
                    onClick={() => handleTechniqueToggle(technique)}
                    className={`px-4 py-3 rounded-lg border-2 transition text-sm font-semibold ${
                      selectedTechniques.includes(technique)
                        ? 'bg-blue-500 text-white border-blue-500 shadow-lg shadow-blue-500/50'
                        : 'bg-black/50 text-gray-300 border-gray-700 hover:border-blue-500/50 hover:text-white'
                    }`}
                  >
                    {technique}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex gap-6">
              <button
                onClick={handleGenerate}
                disabled={!targetColumn || selectedModels.length === 0}
                className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-500 text-white px-10 py-5 rounded-full text-xl font-black hover:from-cyan-600 hover:to-blue-600 transition-all disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed uppercase tracking-wider shadow-2xl shadow-cyan-500/50 hover:shadow-cyan-500/70 hover:scale-105 transform"
              >
                Generate Notebook
              </button>
              <button
                onClick={reset}
                className="px-10 py-5 rounded-full text-xl font-black border-2 border-gray-700 text-gray-300 hover:bg-gray-800 hover:border-gray-600 transition-all uppercase tracking-wider"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
        )}

        {step === 'generating' && (
          <div className="max-w-4xl mx-auto px-6 py-20">
            <div className="bg-gray-900 rounded-2xl shadow-2xl p-12 text-center border border-cyan-500/30">
              <Loader2 className="w-20 h-20 text-cyan-500 mx-auto mb-6 animate-spin" />
              <h2 className="text-3xl font-bold text-white mb-4 uppercase tracking-wide">{progress}</h2>
              <p className="text-gray-400">
                This may take a few minutes as we generate AI-powered theory sections...
              </p>
              <div className="mt-8 max-w-md mx-auto">
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse" style={{ width: '60%' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 'complete' && (
          <div className="max-w-5xl mx-auto px-6 py-12">
            <div className="bg-gray-900 rounded-2xl shadow-2xl p-8 border border-cyan-500/30">
              <div className="text-center mb-8">
                <CheckCircle className="w-20 h-20 text-cyan-500 mx-auto mb-4" />
                <h2 className="text-4xl font-black text-white mb-3 uppercase tracking-wide">
                  Success!
                </h2>
                <p className="text-gray-400 text-lg">
                  Your production-ready ML notebook is ready to download
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mb-8">
                <button
                  onClick={downloadNotebook}
                  className="flex items-center justify-center gap-3 bg-cyan-500 text-white px-8 py-4 rounded-full text-lg font-bold hover:bg-cyan-600 transition uppercase tracking-wide shadow-lg shadow-cyan-500/30"
                >
                  <FileJson className="w-6 h-6" />
                  Download .ipynb
                </button>
                <button
                  onClick={downloadPython}
                  className="flex items-center justify-center gap-3 bg-blue-500 text-white px-8 py-4 rounded-full text-lg font-bold hover:bg-blue-600 transition uppercase tracking-wide shadow-lg shadow-blue-500/30"
                >
                  <FileCode className="w-6 h-6" />
                  Download .py
                </button>
              </div>

              <div className="bg-black/50 rounded-xl p-6 mb-6 border border-cyan-500/20">
                <h3 className="font-bold text-lg mb-4 text-cyan-400 uppercase tracking-wide">What's Included:</h3>
                <ul className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-cyan-500 mt-0.5 flex-shrink-0" />
                    <span>Complete data preprocessing pipeline</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-cyan-500 mt-0.5 flex-shrink-0" />
                    <span>AI-generated theory sections</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-cyan-500 mt-0.5 flex-shrink-0" />
                    <span>Hyperparameter tuning with GridSearchCV</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-cyan-500 mt-0.5 flex-shrink-0" />
                    <span>Comprehensive model evaluation</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-cyan-500 mt-0.5 flex-shrink-0" />
                    <span>Beautiful visualizations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-cyan-500 mt-0.5 flex-shrink-0" />
                    <span>Model deployment preparation</span>
                  </li>
                </ul>
              </div>

              <button
                onClick={reset}
                className="w-full bg-black/50 text-white px-8 py-4 rounded-full text-lg font-bold hover:bg-black/70 transition border-2 border-gray-700 uppercase tracking-wide"
              >
                Generate Another Notebook
              </button>
            </div>
          </div>
        )}

        {step === 'error' && (
          <div className="max-w-4xl mx-auto px-6 py-20">
            <div className="bg-gray-900 rounded-2xl shadow-2xl p-8 border border-red-500/30">
              <div className="text-center mb-6">
                <AlertCircle className="w-20 h-20 text-red-500 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-white mb-3 uppercase tracking-wide">
                  Error
                </h2>
                <p className="text-red-400 mb-6">{error}</p>
              </div>
              <button
                onClick={reset}
                className="w-full bg-cyan-500 text-white px-8 py-4 rounded-full text-lg font-bold hover:bg-cyan-600 transition uppercase tracking-wide"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        <footer className="mt-32 pb-12 text-center">
          <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent mb-8"></div>
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-6 h-6 text-cyan-400/50" />
            <span className="text-xl font-black text-white tracking-tight">GALACTIC<span className="text-cyan-400">ML</span></span>
          </div>
          <p className="text-gray-500 text-xs uppercase tracking-[0.2em] font-bold">Powered by Gemini AI • Built with React</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
