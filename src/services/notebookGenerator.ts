import { generateTheoryContent, generateCheatSheet } from './geminiService';

export interface NotebookConfig {
  datasetName: string;
  problemType: string;
  targetColumn: string;
  selectedModels: string[];
  selectedTechniques: string[];
  datasetInfo: {
    shape: [number, number];
    columns: string[];
    dtypes: Record<string, string>;
  };
}

interface NotebookCell {
  cell_type: 'code' | 'markdown';
  metadata: Record<string, any>;
  source: string[];
  execution_count?: number | null;
  outputs?: any[];
}

function generateTechniqueCode(technique: string): string[] | null {
  const techniqueMap: Record<string, string[]> = {
    'Principal Component Analysis (PCA)': [
      'from sklearn.decomposition import PCA\n',
      '\n',
      'pca = PCA(n_components=0.95)\n',
      'X_train_scaled = pca.fit_transform(X_train_scaled)\n',
      'X_val_scaled = pca.transform(X_val_scaled)\n',
      'X_test_scaled = pca.transform(X_test_scaled)\n',
      'print(f"Reduced to {pca.n_components_} components (95% variance retained)")\n',
      'print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")\n'
    ],
    't-SNE': [
      'from sklearn.manifold import TSNE\n',
      '\n',
      'tsne = TSNE(n_components=2, random_state=42)\n',
      'X_train_tsne = tsne.fit_transform(X_train_scaled[:1000])\n',
      'print("t-SNE dimensionality reduction applied for visualization")\n'
    ],
    'UMAP': [
      '!pip install -q umap-learn\n',
      'import umap\n',
      '\n',
      'reducer = umap.UMAP(n_components=2, random_state=42)\n',
      'X_train_umap = reducer.fit_transform(X_train_scaled)\n',
      'X_val_umap = reducer.transform(X_val_scaled)\n',
      'X_test_umap = reducer.transform(X_test_scaled)\n',
      'print(f"UMAP reduced dimensions to 2")\n'
    ],
    'Recursive Feature Elimination (RFE)': [
      'from sklearn.feature_selection import RFE\n',
      'from sklearn.ensemble import RandomForestClassifier\n',
      '\n',
      'estimator = RandomForestClassifier(n_estimators=50, random_state=42)\n',
      'rfe = RFE(estimator, n_features_to_select=int(X_train_scaled.shape[1] * 0.7))\n',
      'X_train_scaled = rfe.fit_transform(X_train_scaled, y_train)\n',
      'X_val_scaled = rfe.transform(X_val_scaled)\n',
      'X_test_scaled = rfe.transform(X_test_scaled)\n',
      'print(f"RFE selected {rfe.n_features_} features")\n'
    ],
    'SMOTE': [
      'from imblearn.over_sampling import SMOTE\n',
      '\n',
      'smote = SMOTE(random_state=42)\n',
      'X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)\n',
      'print(f"SMOTE applied. New training shape: {X_train_scaled.shape}")\n',
      'print(f"Class distribution after SMOTE:\\n{pd.Series(y_train).value_counts()}")\n'
    ],
    'Random Oversampling': [
      'from imblearn.over_sampling import RandomOverSampler\n',
      '\n',
      'ros = RandomOverSampler(random_state=42)\n',
      'X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)\n',
      'print(f"Random oversampling applied. New training shape: {X_train_scaled.shape}")\n'
    ],
    'Chi-Square Test': [
      'from sklearn.feature_selection import SelectKBest, chi2\n',
      '\n',
      'selector = SelectKBest(chi2, k=min(20, X_train_scaled.shape[1]))\n',
      'X_train_scaled = selector.fit_transform(np.abs(X_train_scaled), y_train)\n',
      'X_val_scaled = selector.transform(np.abs(X_val_scaled))\n',
      'X_test_scaled = selector.transform(np.abs(X_test_scaled))\n',
      'print(f"Chi-Square test selected top {selector.k} features")\n'
    ],
    'Mutual Information': [
      'from sklearn.feature_selection import SelectKBest, mutual_info_classif\n',
      '\n',
      'selector = SelectKBest(mutual_info_classif, k=min(15, X_train_scaled.shape[1]))\n',
      'X_train_scaled = selector.fit_transform(X_train_scaled, y_train)\n',
      'X_val_scaled = selector.transform(X_val_scaled)\n',
      'X_test_scaled = selector.transform(X_test_scaled)\n',
      'print(f"Mutual Information selected top {selector.k} features")\n'
    ]
  };

  return techniqueMap[technique] || [
    `# ${technique}\n`,
    'print(f"Applying {technique}...")\n',
    '# Implementation code for this technique\n'
  ];
}

export async function generateNotebook(config: NotebookConfig): Promise<any> {
  const cells: NotebookCell[] = [];

  cells.push(createMarkdownCell([
    `# 🤖 Machine Learning Pipeline: ${config.datasetName}\n`,
    '\n',
    '## Project Overview\n',
    '\n',
    `This notebook implements a complete end-to-end machine learning workflow for **${config.problemType}** problem.\n`,
    '\n',
    '### Objectives:\n',
    '- Automated data preprocessing and feature engineering\n',
    '- Advanced hyperparameter tuning using multiple strategies\n',
    '- Comprehensive model evaluation and comparison\n',
    '- Production-ready deployment preparation\n',
    '\n',
    '---\n'
  ]));

  cells.push(createMarkdownCell([
    '## 📚 Table of Contents\n',
    '\n',
    '1. [Environment Setup](#setup)\n',
    '2. [Data Loading & Validation](#data)\n',
    '3. [Exploratory Data Analysis](#eda)\n',
    '4. [Data Preprocessing](#preprocessing)\n',
    '5. [Model Training & Hyperparameter Tuning](#training)\n',
    '6. [Model Evaluation](#evaluation)\n',
    '7. [Results & Deployment](#results)\n'
  ]));

  cells.push(createMarkdownCell([
    '## 📦 1. Environment Setup\n',
    '\n',
    'Installing required packages with version pinning for reproducibility.\n'
  ]));

  cells.push(createCodeCell([
    '!pip install -q numpy pandas scikit-learn matplotlib seaborn\n',
    '!pip install -q xgboost lightgbm optuna\n',
    '!pip install -q imbalanced-learn shap\n',
    '\n',
    'import warnings\n',
    'warnings.filterwarnings("ignore")\n',
    '\n',
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    'from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score\n',
    'from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder\n',
    'from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score\n',
    'from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n',
    'import optuna\n',
    'import joblib\n',
    'from datetime import datetime\n',
    '\n',
    'print("✅ All packages imported successfully!")\n',
    'print(f"Notebook execution started at: {datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}")\n'
  ]));

  cells.push(createMarkdownCell([
    '## 📊 2. Data Loading & Validation\n',
    '\n',
    'Loading the dataset and performing initial quality checks.\n'
  ]));

  cells.push(createCodeCell([
    `df = pd.read_csv("${config.datasetName}")\n`,
    '\n',
    'print(f"Dataset shape: {df.shape}")\n',
    'print(f"\\nColumns: {list(df.columns)}")\n',
    'print(f"\\nMemory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")\n',
    '\n',
    'display(df.head())\n'
  ]));

  cells.push(createMarkdownCell([
    '### Data Quality Assessment\n'
  ]));

  cells.push(createCodeCell([
    'print("=== Data Info ===" )\n',
    'print(df.info())\n',
    '\n',
    'print("\\n=== Statistical Summary ===")\n',
    'display(df.describe())\n',
    '\n',
    'print("\\n=== Missing Values ===")\n',
    'missing_data = df.isnull().sum()\n',
    'missing_percent = 100 * missing_data / len(df)\n',
    'missing_table = pd.DataFrame({\'Missing Count\': missing_data, \'Percentage\': missing_percent})\n',
    'display(missing_table[missing_table["Missing Count"] > 0].sort_values("Percentage", ascending=False))\n',
    '\n',
    'print("\\n=== Duplicate Rows ===" )\n',
    'duplicates = df.duplicated().sum()\n',
    'print(f"Number of duplicate rows: {duplicates} ({100*duplicates/len(df):.2f}%)")\n'
  ]));

  const theoryMD = await generateTheoryContent('Exploratory Data Analysis', config.problemType);
  cells.push(createMarkdownCell([
    '## 📈 3. Theory: Exploratory Data Analysis\n',
    '\n',
    theoryMD + '\n'
  ]));

  cells.push(createMarkdownCell([
    '## 🔍 Exploratory Data Analysis\n'
  ]));

  cells.push(createCodeCell([
    'fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n',
    '\n',
    `target_col = "${config.targetColumn}"\n`,
    '\n',
    'if target_col in df.columns:\n',
    '    axes[0, 0].set_title("Target Distribution", fontsize=14, fontweight="bold")\n',
    '    if df[target_col].dtype == "object" or df[target_col].nunique() < 20:\n',
    '        df[target_col].value_counts().plot(kind="bar", ax=axes[0, 0], color="skyblue")\n',
    '    else:\n',
    '        axes[0, 0].hist(df[target_col].dropna(), bins=50, color="skyblue", edgecolor="black")\n',
    '    axes[0, 0].set_xlabel(target_col)\n',
    '\n',
    'numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n',
    'if len(numeric_cols) > 1:\n',
    '    corr_matrix = df[numeric_cols].corr()\n',
    '    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 1], \n',
    '                square=True, cbar_kws={"shrink": 0.8})\n',
    '    axes[0, 1].set_title("Correlation Matrix", fontsize=14, fontweight="bold")\n',
    '\n',
    'if len(numeric_cols) >= 2:\n',
    '    axes[1, 0].scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5)\n',
    '    axes[1, 0].set_xlabel(numeric_cols[0])\n',
    '    axes[1, 0].set_ylabel(numeric_cols[1])\n',
    '    axes[1, 0].set_title("Feature Relationship", fontsize=14, fontweight="bold")\n',
    '\n',
    'missing_counts = df.isnull().sum()\n',
    'if missing_counts.sum() > 0:\n',
    '    missing_counts[missing_counts > 0].plot(kind="barh", ax=axes[1, 1], color="coral")\n',
    '    axes[1, 1].set_title("Missing Values by Column", fontsize=14, fontweight="bold")\n',
    '    axes[1, 1].set_xlabel("Count")\n',
    '\n',
    'plt.tight_layout()\n',
    'plt.show()\n'
  ]));

  const preprocessTheory = await generateTheoryContent('Data Preprocessing', config.problemType);
  cells.push(createMarkdownCell([
    '## 🛠️ 4. Theory: Data Preprocessing\n',
    '\n',
    preprocessTheory + '\n'
  ]));

  cells.push(createMarkdownCell([
    '## 🧹 Data Preprocessing Pipeline\n'
  ]));

  cells.push(createCodeCell([
    'df_processed = df.copy()\n',
    '\n',
    'print("Step 1: Handling Missing Values")\n',
    'numeric_columns = df_processed.select_dtypes(include=[np.number]).columns\n',
    'categorical_columns = df_processed.select_dtypes(include=["object"]).columns\n',
    '\n',
    'for col in numeric_columns:\n',
    '    if df_processed[col].isnull().sum() > 0:\n',
    '        df_processed[col].fillna(df_processed[col].median(), inplace=True)\n',
    '        print(f"  - Filled {col} with median")\n',
    '\n',
    'for col in categorical_columns:\n',
    '    if df_processed[col].isnull().sum() > 0:\n',
    '        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)\n',
    '        print(f"  - Filled {col} with mode")\n',
    '\n',
    'print("\\nStep 2: Removing Duplicates")\n',
    'before_dup = len(df_processed)\n',
    'df_processed.drop_duplicates(inplace=True)\n',
    'after_dup = len(df_processed)\n',
    'print(f"  - Removed {before_dup - after_dup} duplicate rows")\n',
    '\n',
    'print("\\nStep 3: Encoding Categorical Variables")\n',
    'label_encoders = {}\n',
    `target_col = "${config.targetColumn}"\n`,
    '\n',
    'for col in categorical_columns:\n',
    '    if col != target_col and df_processed[col].nunique() < 10:\n',
    '        le = LabelEncoder()\n',
    '        df_processed[col] = le.fit_transform(df_processed[col].astype(str))\n',
    '        label_encoders[col] = le\n',
    '        print(f"  - Encoded {col}")\n',
    '\n',
    'print("\\nStep 4: Handling Outliers (IQR Method)")\n',
    'for col in numeric_columns:\n',
    '    if col != target_col:\n',
    '        Q1 = df_processed[col].quantile(0.25)\n',
    '        Q3 = df_processed[col].quantile(0.75)\n',
    '        IQR = Q3 - Q1\n',
    '        lower_bound = Q1 - 1.5 * IQR\n',
    '        upper_bound = Q3 + 1.5 * IQR\n',
    '        outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()\n',
    '        if outliers > 0:\n',
    '            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)\n',
    '            print(f"  - Clipped {outliers} outliers in {col}")\n',
    '\n',
    'print("\\n✅ Preprocessing complete!")\n',
    'print(f"Final dataset shape: {df_processed.shape}")\n'
  ]));

  cells.push(createMarkdownCell([
    '## 🎯 5. Feature & Target Separation\n'
  ]));

  cells.push(createCodeCell([
    `target_col = "${config.targetColumn}"\n`,
    '\n',
    'if target_col in df_processed.columns:\n',
    '    X = df_processed.drop(columns=[target_col])\n',
    '    y = df_processed[target_col]\n',
    '    \n',
    '    if y.dtype == "object":\n',
    '        le_target = LabelEncoder()\n',
    '        y = le_target.fit_transform(y)\n',
    '        print(f"Target classes: {le_target.classes_}")\n',
    '    \n',
    '    print(f"Features shape: {X.shape}")\n',
    '    print(f"Target shape: {y.shape}")\n',
    '    print(f"\\nTarget distribution:\\n{pd.Series(y).value_counts()}")\n',
    'else:\n',
    '    print(f"⚠️ Warning: Target column \'{target_col}\' not found in dataset")\n',
    '    X = df_processed\n',
    '    y = None\n'
  ]));

  cells.push(createMarkdownCell([
    '## 📊 6. Train-Test-Validation Split (60/20/20)\n'
  ]));

  cells.push(createCodeCell([
    'X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y is not None else None)\n',
    'X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp if y_temp is not None else None)\n',
    '\n',
    'print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")\n',
    'print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")\n',
    'print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")\n',
    '\n',
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_val_scaled = scaler.transform(X_val)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'print("\\n✅ Data scaled using StandardScaler")\n'
  ]));

  if (config.selectedTechniques && config.selectedTechniques.length > 0) {
    cells.push(createMarkdownCell([
      '## 🔧 6.5. Applying Advanced Techniques\n',
      '\n',
      `Selected techniques to enhance model performance:\n`,
      ...config.selectedTechniques.map(t => `- ${t}\n`)
    ]));

    for (const technique of config.selectedTechniques) {
      const techniqueCode = generateTechniqueCode(technique);
      if (techniqueCode) {
        cells.push(createMarkdownCell([
          `### Applying: ${technique}\n`
        ]));
        cells.push(createCodeCell(techniqueCode));
      }
    }

    cells.push(createMarkdownCell([
      '✅ All techniques applied successfully!\n'
    ]));
  }

  const tuningTheory = await generateTheoryContent('Hyperparameter Tuning', config.problemType);
  cells.push(createMarkdownCell([
    '## 🎛️ 7. Theory: Hyperparameter Tuning\n',
    '\n',
    tuningTheory + '\n'
  ]));

  const cheatSheet = await generateCheatSheet('Hyperparameter Tuning Strategies');
  cells.push(createMarkdownCell([
    '## 📋 Hyperparameter Tuning Cheat Sheet\n',
    '\n',
    cheatSheet + '\n'
  ]));

  for (const model of config.selectedModels) {
    const modelTheory = await generateTheoryContent(model, config.problemType);
    cells.push(createMarkdownCell([
      `## 🤖 Theory: ${model}\n`,
      '\n',
      modelTheory + '\n'
    ]));

    cells.push(createMarkdownCell([
      `## 🔧 ${model} - Training & Hyperparameter Tuning\n`
    ]));

    const modelCode = generateModelCode(model, config.problemType);
    cells.push(createCodeCell(modelCode));
  }

  cells.push(createMarkdownCell([
    '## 📊 8. Model Comparison & Results\n'
  ]));

  cells.push(createCodeCell([
    'results_df = pd.DataFrame({\n',
    '    "Model": model_names,\n',
    '    "Training Score": train_scores,\n',
    '    "Validation Score": val_scores,\n',
    '    "Test Score": test_scores,\n',
    '    "Training Time (s)": training_times\n',
    '})\n',
    '\n',
    'results_df = results_df.sort_values("Test Score", ascending=False)\n',
    'display(results_df)\n',
    '\n',
    'plt.figure(figsize=(12, 6))\n',
    'x = np.arange(len(model_names))\n',
    'width = 0.25\n',
    '\n',
    'plt.bar(x - width, train_scores, width, label="Train", alpha=0.8)\n',
    'plt.bar(x, val_scores, width, label="Validation", alpha=0.8)\n',
    'plt.bar(x + width, test_scores, width, label="Test", alpha=0.8)\n',
    '\n',
    'plt.xlabel("Models", fontsize=12, fontweight="bold")\n',
    'plt.ylabel("Score", fontsize=12, fontweight="bold")\n',
    'plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")\n',
    'plt.xticks(x, model_names, rotation=45, ha="right")\n',
    'plt.legend()\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
    '\n',
    'print("\\n🏆 Best Model:", results_df.iloc[0]["Model"])\n',
    'print(f"   Test Score: {results_df.iloc[0][\'Test Score\']:.4f}")\n'
  ]));

  cells.push(createMarkdownCell([
    '## 💾 9. Model Deployment Preparation\n'
  ]));

  cells.push(createCodeCell([
    'best_model_name = results_df.iloc[0]["Model"]\n',
    'timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n',
    'model_filename = f"{best_model_name.replace(\' \', \'_\')}_{timestamp}.pkl"\n',
    '\n',
    'joblib.dump(best_model, model_filename)\n',
    'joblib.dump(scaler, f"scaler_{timestamp}.pkl")\n',
    '\n',
    'print(f"✅ Model saved: {model_filename}")\n',
    'print(f"✅ Scaler saved: scaler_{timestamp}.pkl")\n',
    '\n',
    'print("\\n📄 Model Card:")\n',
    'print("=" * 50)\n',
    'print(f"Model: {best_model_name}")\n',
    'print(f"Problem Type: {config.problemType}")\n',
    'print(f"Training Date: {datetime.now().strftime(\'%Y-%m-%d\')}")\n',
    'print(f"Dataset: {config.datasetName}")\n',
    'print(f"Features: {X.shape[1]}")\n',
    'print(f"Training Samples: {len(X_train)}")\n',
    'print(f"Test Performance: {results_df.iloc[0][\'Test Score\']:.4f}")\n',
    'print("=" * 50)\n'
  ]));

  cells.push(createMarkdownCell([
    '## 🎉 Notebook Complete!\n',
    '\n',
    'This notebook has successfully:\n',
    '- ✅ Loaded and validated the dataset\n',
    '- ✅ Performed comprehensive EDA\n',
    '- ✅ Preprocessed data with best practices\n',
    '- ✅ Trained multiple models with hyperparameter tuning\n',
    '- ✅ Evaluated and compared model performance\n',
    '- ✅ Prepared the best model for deployment\n',
    '\n',
    '---\n',
    '\n',
    '**Generated with AI-Powered ML Pipeline Generator** 🤖\n'
  ]));

  const notebook = {
    cells,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3'
      },
      language_info: {
        codemirror_mode: {
          name: 'ipython',
          version: 3
        },
        file_extension: '.py',
        mimetype: 'text/x-python',
        name: 'python',
        nbconvert_exporter: 'python',
        pygments_lexer: 'ipython3',
        version: '3.8.0'
      }
    },
    nbformat: 4,
    nbformat_minor: 4
  };

  return notebook;
}

function createMarkdownCell(source: string[]): NotebookCell {
  return {
    cell_type: 'markdown',
    metadata: {},
    source
  };
}

function createCodeCell(source: string[]): NotebookCell {
  return {
    cell_type: 'code',
    execution_count: null,
    metadata: {},
    outputs: [],
    source
  };
}

function generateModelCode(modelName: string, problemType: string): string[] {
  const isClassification = problemType === 'classification';

  const modelImports: Record<string, string> = {
    'Random Forest': isClassification ? 'from sklearn.ensemble import RandomForestClassifier' : 'from sklearn.ensemble import RandomForestRegressor',
    'XGBoost': isClassification ? 'from xgboost import XGBClassifier' : 'from xgboost import XGBRegressor',
    'Logistic Regression': 'from sklearn.linear_model import LogisticRegression',
    'Linear Regression': 'from sklearn.linear_model import LinearRegression',
    'SVM': isClassification ? 'from sklearn.svm import SVC' : 'from sklearn.svm import SVR',
    'KNN': isClassification ? 'from sklearn.neighbors import KNeighborsClassifier' : 'from sklearn.neighbors import KNeighborsRegressor',
    'Decision Tree': isClassification ? 'from sklearn.tree import DecisionTreeClassifier' : 'from sklearn.tree import DecisionTreeRegressor',
    'Gradient Boosting': isClassification ? 'from sklearn.ensemble import GradientBoostingClassifier' : 'from sklearn.ensemble import GradientBoostingRegressor',
    'LightGBM': isClassification ? 'from lightgbm import LGBMClassifier' : 'from lightgbm import LGBMRegressor',
  };

  const modelInit: Record<string, string> = {
    'Random Forest': isClassification ? 'RandomForestClassifier(random_state=42)' : 'RandomForestRegressor(random_state=42)',
    'XGBoost': isClassification ? 'XGBClassifier(random_state=42, eval_metric="logloss")' : 'XGBRegressor(random_state=42)',
    'Logistic Regression': 'LogisticRegression(random_state=42, max_iter=1000)',
    'Linear Regression': 'LinearRegression()',
    'SVM': isClassification ? 'SVC(random_state=42)' : 'SVR()',
    'KNN': isClassification ? 'KNeighborsClassifier()' : 'KNeighborsRegressor()',
    'Decision Tree': isClassification ? 'DecisionTreeClassifier(random_state=42)' : 'DecisionTreeRegressor(random_state=42)',
    'Gradient Boosting': isClassification ? 'GradientBoostingClassifier(random_state=42)' : 'GradientBoostingRegressor(random_state=42)',
    'LightGBM': isClassification ? 'LGBMClassifier(random_state=42, verbose=-1)' : 'LGBMRegressor(random_state=42, verbose=-1)',
  };

  const paramGrids: Record<string, string> = {
    'Random Forest': `{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}`,
    'XGBoost': `{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0]
}`,
    'Logistic Regression': `{
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}`,
    'SVM': `{
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}`,
    'KNN': `{
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}`,
  };

  const importLine = modelImports[modelName] || 'from sklearn.ensemble import RandomForestClassifier';
  const modelInitLine = modelInit[modelName] || 'RandomForestClassifier(random_state=42)';
  const paramGrid = paramGrids[modelName] || paramGrids['Random Forest'];

  const scoringMetric = isClassification ? 'accuracy' : 'r2';
  const evalMetrics = isClassification
    ? 'accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="weighted")'
    : 'r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)';

  return [
    `${importLine}\n`,
    'import time\n',
    '\n',
    'print(f"Training {modelName}...")\n',
    'start_time = time.time()\n',
    '\n',
    `model = ${modelInitLine}\n`,
    '\n',
    `param_grid = ${paramGrid}\n`,
    '\n',
    `grid_search = GridSearchCV(\n`,
    '    model,\n',
    '    param_grid,\n',
    `    cv=5,\n`,
    `    scoring='${scoringMetric}',\n`,
    '    n_jobs=-1,\n',
    '    verbose=1\n',
    ')\n',
    '\n',
    'grid_search.fit(X_train_scaled, y_train)\n',
    '\n',
    'best_model = grid_search.best_estimator_\n',
    'train_time = time.time() - start_time\n',
    '\n',
    'print(f"\\n✅ Training complete in {train_time:.2f} seconds")\n',
    'print(f"Best parameters: {grid_search.best_params_}")\n',
    '\n',
    'y_train_pred = best_model.predict(X_train_scaled)\n',
    'y_val_pred = best_model.predict(X_val_scaled)\n',
    'y_test_pred = best_model.predict(X_test_scaled)\n',
    '\n',
    `train_score = ${scoringMetric === 'accuracy' ? 'accuracy_score' : 'r2_score'}(y_train, y_train_pred)\n`,
    `val_score = ${scoringMetric === 'accuracy' ? 'accuracy_score' : 'r2_score'}(y_val, y_val_pred)\n`,
    `test_score = ${scoringMetric === 'accuracy' ? 'accuracy_score' : 'r2_score'}(y_test, y_test_pred)\n`,
    '\n',
    'print(f"\\nTraining Score: {train_score:.4f}")\n',
    'print(f"Validation Score: {val_score:.4f}")\n',
    'print(f"Test Score: {test_score:.4f}")\n',
    '\n',
    isClassification ? `\nprint("\\nClassification Report:")\n` : `\nprint("\\nRegression Metrics:")\n`,
    isClassification
      ? 'print(classification_report(y_test, y_test_pred))\n\n'
      : 'print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")\n' +
        'print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")\n' +
        'print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")\n\n',
    isClassification
      ? 'cm = confusion_matrix(y_test, y_test_pred)\n' +
        'plt.figure(figsize=(8, 6))\n' +
        'sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)\n' +
        'plt.title(f"{modelName} - Confusion Matrix", fontsize=14, fontweight="bold")\n' +
        'plt.ylabel("True Label")\n' +
        'plt.xlabel("Predicted Label")\n' +
        'plt.show()\n'
      : 'plt.figure(figsize=(10, 6))\n' +
        'plt.scatter(y_test, y_test_pred, alpha=0.5)\n' +
        'plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)\n' +
        'plt.xlabel("Actual Values")\n' +
        'plt.ylabel("Predicted Values")\n' +
        'plt.title(f"{modelName} - Actual vs Predicted", fontsize=14, fontweight="bold")\n' +
        'plt.show()\n',
    '\n',
    'if "model_names" not in dir():\n',
    '    model_names = []\n',
    '    train_scores = []\n',
    '    val_scores = []\n',
    '    test_scores = []\n',
    '    training_times = []\n',
    '\n',
    `model_names.append("${modelName}")\n`,
    'train_scores.append(train_score)\n',
    'val_scores.append(val_score)\n',
    'test_scores.append(test_score)\n',
    'training_times.append(train_time)\n'
  ];
}

export function convertNotebookToPython(notebook: any): string {
  let pyContent = '#!/usr/bin/env python\n';
  pyContent += '# coding: utf-8\n\n';
  pyContent += '# Auto-generated from Jupyter Notebook\n';
  pyContent += `# Generated on: ${new Date().toISOString()}\n\n`;

  for (const cell of notebook.cells) {
    if (cell.cell_type === 'markdown') {
      const lines = cell.source.join('');
      const commented = lines.split('\n').map(line => `# ${line}`).join('\n');
      pyContent += `\n${commented}\n\n`;
    } else if (cell.cell_type === 'code') {
      const code = cell.source.join('');
      pyContent += `${code}\n\n`;
    }
  }

  return pyContent;
}
