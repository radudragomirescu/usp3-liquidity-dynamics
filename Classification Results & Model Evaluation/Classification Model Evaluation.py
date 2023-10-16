# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
# Classification methods
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# Classification model evaluation
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
# Dataset shift inspection 
from scipy.stats import gaussian_kde
from scipy.stats import entropy


""" Classification Model Evaluation Setup """

# Create DataFrames for Pools 5 and 30
usdc_weth_5 = pd.read_csv("usdc_weth_5_final.csv")
usdc_weth_30 = pd.read_csv("usdc_weth_30_final.csv")

# Define seed number to control randomness
rng=2023

# Generates training/validation and test data splits for classification tasks
def process_dataframe_clf(dataframe, y_column):
    X = dataframe.drop(columns=['date', 'block_number', 'pool_name', 'price', 'next_type_main',
                                'next_type_other', 'next_mint_time_main', 'next_burn_time_main',
                                'next_mint_time_other', 'next_burn_time_other'])
    # One-hot encode the categorical features
    X = pd.get_dummies(X, prefix = ['type', 'liquidity'], drop_first = True)
    
    y = dataframe[y_column]

    
    # No shuffle because of our time series setting
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, shuffle = False, test_size = 0.2)
    
    return X_train_val, X_test, y_train_val, y_test

# Defines a StandardScaler which only applies to numerical features
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_scalers = {}

    def fit(self, X, y=None):
        self.column_scalers = {}
        for col in X.columns:
            if len(np.unique(X[col])) > 2:  # Check if the column is non-binary
                self.column_scalers[col] = StandardScaler()
                self.column_scalers[col].fit(X[[col]])
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, scaler in self.column_scalers.items():
            X_copy[col] = scaler.transform(X_copy[[col]])
        return X_copy

# Defines a PowerTransformer which only applies to numerical features
class CustomPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='yeo-johnson'):
        self.method = method
        self.column_transformers = {}

    def fit(self, X, y=None):
        self.column_transformers = {}
        for col in X.columns:
            if len(np.unique(X[col])) > 2:  # Check if the column is non-binary
                self.column_transformers[col] = PowerTransformer(method=self.method)
                self.column_transformers[col].fit(X[[col]])
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, transformer in self.column_transformers.items():
            X_copy[col] = transformer.transform(X_copy[[col]])
        return X_copy

# Create training/validation and test splits for all classification targets
X_train_val_5_main, X_test_5_main, y_train_val_5_main, y_test_5_main = process_dataframe_clf(usdc_weth_5, "next_type_main")
X_train_val_5_other, X_test_5_other, y_train_val_5_other, y_test_5_other = process_dataframe_clf(usdc_weth_5, "next_type_other")
X_train_val_30_main, X_test_30_main, y_train_val_30_main, y_test_30_main = process_dataframe_clf(usdc_weth_30, "next_type_main")
X_train_val_30_other, X_test_30_other, y_train_val_30_other, y_test_30_other = process_dataframe_clf(usdc_weth_30, "next_type_other")

# Load in the best estimator for each target
model_5_main = joblib.load("ridge_5_main_final.joblib")
model_5_other = joblib.load("decision_5_other_final.joblib")
model_30_main = joblib.load("lgbm_30_main_final.joblib")
model_30_other = joblib.load("knn_30_other_final.joblib")

# Make predictions on the test split of each target with the respective best estimator
y_pred_5_main = model_5_main.predict(X_test_5_main)
y_pred_5_other = model_5_other.predict(X_test_5_other)
y_pred_30_main = model_30_main.predict(X_test_30_main)
y_pred_30_other = model_30_other.predict(X_test_30_other)

# Make predictions on the training/validation split of each target with the respective best estimator
y_train_pred_5_main = model_5_main.predict(X_train_val_5_main)
y_train_pred_5_other = model_5_other.predict(X_train_val_5_other)
y_train_pred_30_main = model_30_main.predict(X_train_val_30_main)
y_train_pred_30_other = model_30_other.predict(X_train_val_30_other)


# Helper function to generate baseline persistence predictions
def get_pers_other(row):
    if row['b1_mint_other'] < row['b1_burn_other']:
        return 'mint'
    else:
        return 'burn'

# Make predictions on the test split with the baseline persistence model for each target
y_pers_5_main =  X_test_5_main['type_mint'].apply(lambda x: 'mint' if x == 1 else 'burn').values
y_pers_5_other = X_test_5_main.apply(get_pers_other, axis=1).values
y_pers_30_main = X_test_30_main['type_mint'].apply(lambda x: 'mint' if x == 1 else 'burn').values
y_pers_30_other = X_test_30_main.apply(get_pers_other, axis=1).values


""" Classification Model Evaluation Plots & Tables """

# Tables 6.1 and 6.4 (Persistence Accuracy)
print(accuracy_score(y_test_5_main, y_pers_5_main))
print(accuracy_score(y_test_5_other, y_pers_5_other))
print(accuracy_score(y_test_30_main, y_pers_30_main))
print(accuracy_score(y_test_30_other, y_pers_30_other))

# Tables 6.1 and 6.4 (Best Estimator Accuracy)
print(accuracy_score(y_test_5_main, y_pred_5_main))
print(accuracy_score(y_test_5_other, y_pred_5_other))
print(accuracy_score(y_test_30_main, y_pred_30_main))
print(accuracy_score(y_test_30_other, y_pred_30_other))

# Plots normalized and unnormalized confusion matrices for the input test split predictions and target
def plot_confusion_matrices(y_test, y_pred, labels=["burn", "mint"], title=None):
    # Calculate unnormalized confusion matrix
    confusion_unnormalized = confusion_matrix(y_test, y_pred)

    # Calculate normalized confusion matrix
    confusion_normalized = confusion_matrix(y_test, y_pred, normalize='true')

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi = 300)

    # Plot unnormalized confusion matrix
    disp_unnormalized = ConfusionMatrixDisplay(confusion_matrix=confusion_unnormalized, display_labels=labels)
    disp_unnormalized.plot(ax=axes[0])
    axes[0].set_title('Unnormalized Confusion Matrix', fontsize = 18)
    
    disp_unnormalized.ax_.tick_params(axis='both', labelsize=14)
    
    axes[0].set_xlabel('Predicted Label', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=14)

    # Plot normalized confusion matrix
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=confusion_normalized, display_labels=labels)
    disp_normalized.plot(ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix', fontsize = 18)
    
    disp_normalized.ax_.tick_params(axis='both', labelsize=14)
    
    axes[1].set_xlabel('Predicted Label', fontsize=14)
    axes[1].set_ylabel('True Label', fontsize=14)
    
    if title:
        plt.suptitle(title, fontsize=28, y=1.05)

    plt.tight_layout()
    plt.show()

# Figure 6.2
plot_confusion_matrices(y_test_5_main, y_pred_5_main, title = "Confusion Matrices for next_type_main (Main Pool: 5)")

# Figure 6.5
plot_confusion_matrices(y_test_5_other, y_pred_5_other, title = "Confusion Matrices for next_type_other (Main Pool: 5)")

# Figure 6.8
plot_confusion_matrices(y_test_30_main, y_pred_30_main, title = "Confusion Matrices for next_type_main (Main Pool: 30)")

# Figure 6.10
plot_confusion_matrices(y_test_30_other, y_pred_30_other, title = "Confusion Matrices for next_type_other (Main Pool: 30)")

# Prints key classification metrics for the input test split predictions and target
def print_classification_report(y_test, y_pred, labels=["burn", "mint"]):
    report = classification_report(y_test, y_pred, labels=labels, digits=4)
    print(report)

# Table 6.2 
print_classification_report(y_test_5_main, y_pred_5_main)

# Table 6.3 
print_classification_report(y_test_5_other, y_pred_5_other)

# Table 6.5 
print_classification_report(y_test_30_main, y_pred_30_main)

# Table 6.6
print_classification_report(y_test_30_other, y_pred_30_other)

# Plots ROC curve and measures AUC for the predicting mints with the input estimator and test data
def plot_roc_curve(estimator, X_test, y_test, title='Receiver Operating Characteristic (ROC) Curve'):
    # Predict probabilities using the estimator
    if hasattr(estimator, 'predict_proba'):
        y_score = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, 'decision_function'):
        y_score = estimator.decision_function(X_test)
    else:
        raise AttributeError("The estimator does not have predict_proba or decision_function")

    # Compute ROC curve and AUC for predicting mints
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label='mint')
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve with AUC and chance comparison
    plt.figure(figsize=(6, 6), dpi = 300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='mint vs. burn (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=20, y = 1.02) 
    plt.legend(loc='lower right', fontsize=14)
    plt.show()

# Figure 6.3a
plot_roc_curve(model_5_main, X_test_5_main, y_test_5_main, title="ROC and AUC for next_type_main (Main Pool: 5)")

# Figure 6.3b
plot_roc_curve(model_5_other, X_test_5_other, y_test_5_other, title="ROC and AUC for next_type_other (Main Pool: 5)")

# Figure 6.3c
plot_roc_curve(model_30_main, X_test_30_main, y_test_30_main, "ROC and AUC for next_type_main (Main Pool: 30)")

# Figure 6.3d
plot_roc_curve(model_30_other, X_test_30_other, y_test_30_other, "ROC and AUC for next_type_other (Main Pool: 30)")

# Plots training/validation and test marginal distributions for the input target and predictions
def plot_class_dists(y_test_pred, y_test, y_train_pred, y_train, title='Class Distribution'):
    # Create dataframes for training/validation and test splits
    categories = ['mint', 'burn']
    train_counts = [sum(y_train == 'mint'), sum(y_train == 'burn')]
    train_pred_counts = [sum(y_train_pred == 'mint'), sum(y_train_pred == 'burn')]
    test_counts = [sum(y_test == 'mint'), sum(y_test == 'burn')]
    test_pred_counts = [sum(y_test_pred == 'mint'), sum(y_test_pred == 'burn')]

    bar_width = 0.35

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi = 300)
    fig.suptitle(title, fontsize=24, y=1.08)

    # Calculate the positions for bars
    train_positions = np.arange(len(categories))
    train_pred_positions = train_positions + bar_width
    test_positions = train_positions
    test_pred_positions = train_pred_positions

   # Plot class distribution for training/validation data (left subplot)
    train_bars = axes[0].bar(train_positions, train_counts, width=bar_width, label='y_train', color='green')
    train_pred_bars = axes[0].bar(train_pred_positions, train_pred_counts, width=bar_width, label='y_train_pred', color='red')
    axes[0].set_title('Training/Validation Split Target Distributions', fontsize = 14)
    axes[0].set_xlabel('Class', fontsize = 14)
    axes[0].set_ylabel('Count', fontsize = 14)
    axes[0].set_xticks(train_positions + bar_width / 2)
    axes[0].set_xticklabels(categories, fontsize = 12)
    axes[0].legend(title='Data', fontsize = 12)


    # Plot class distribution for test data (right subplot)
    test_bars = axes[1].bar(test_positions, test_counts, width=bar_width, label='y_test', color='green')
    test_pred_bars = axes[1].bar(test_pred_positions, test_pred_counts, width=bar_width, label='y_test_pred', color='red')
    axes[1].set_title('Test Split Target Distributions', fontsize = 14)
    axes[1].set_xlabel('Class', fontsize = 14)
    axes[1].set_ylabel('Count', fontsize = 14)
    axes[1].set_xticks(test_positions + bar_width / 2)
    axes[1].set_xticklabels(categories, fontsize = 12)
    axes[1].legend(title='Data', fontsize = 12)

    # Add counts on top of each bar for training/validation data
    for bar, count in zip(train_bars + train_pred_bars, train_counts + train_pred_counts):
        axes[0].annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', weight='bold')

    # Add counts on top of each bar for test data
    for bar, count in zip(test_bars + test_pred_bars, test_counts + test_pred_counts):
        axes[1].annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', weight='bold')

    axes[0].set_ylim(0, len(y_train) + 30)
    axes[1].set_ylim(0, len(y_test) + 30)
    
    plt.show()

# Figure 6.1
plot_class_dists(y_pred_5_main, y_test_5_main, y_train_pred_5_main, y_train_val_5_main, title="Marginal Distributions of next_type_main (Main Pool: 5)")

# Figure 6.4 
plot_class_dists(y_pred_5_other, y_test_5_other, y_train_pred_5_other, y_train_val_5_other, title="Marginal Distributions of next_type_other (Main Pool: 5)")

# Figure 6.7
plot_class_dists(y_pred_30_main, y_test_30_main, y_train_pred_30_main, y_train_val_30_main, title="Marginal Distributions of next_type_main (Main Pool: 30)")

# Figure 6.9
plot_class_dists(y_pred_30_other, y_test_30_other, y_train_pred_30_other, y_train_val_30_other, title="Marginal Distributions of next_type_other (Main Pool: 30)")

# Plot style for feature importance plots
sns.set_style("darkgrid")

# Plots top 10 features by mean feature importance over 50 iterations on the test split for the input classifier
def plot_permutation_importance(model, X_test, y_test, title="Permutation Importance", top_n=10, n_repeats=50, rng=rng):
   
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, scoring='accuracy',
                                             random_state=rng)

    # Get the indices of the top N features based on their mean importance scores
    top_indices = perm_importance.importances_mean.argsort()[-top_n:]

    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({'Feature': X_test.columns[top_indices],
                                   'Mean_Importance': perm_importance.importances_mean[top_indices],
                                   'Std_Importance': perm_importance.importances_std[top_indices]})

    # Sort by mean importance (descending order)
    importance_df = importance_df.sort_values(by='Mean_Importance', ascending=False)

    # Create a barplot 
    plt.figure(figsize=(8, 5), dpi=300)
    ax = sns.barplot(data=importance_df, x='Mean_Importance', y='Feature', palette='viridis')
    for i, (_, row) in enumerate(importance_df.iterrows()):
        ax.text(row['Mean_Importance'], i, f" ± {row['Std_Importance']:.4f}", va='center', ha='left', color='black')
    plt.title(title, y = 1.02, fontsize = 16)
    plt.xlabel('Mean Decrease in Accuracy (± SD)', fontsize = 12)
    plt.ylabel('')  # Remove the y-label
    plt.xlim(right=max(importance_df['Mean_Importance']) + 0.005) 
    
    ax.tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    plt.show()

# Figure 6.6a
plot_permutation_importance(model_5_main, X_test_5_main, y_test_5_main,
                            title = 'Feature Permutation Importance for Classifying next_type_main (Main Pool: 5)')

# Figure 6.6b
plot_permutation_importance(model_5_other, X_test_5_other, y_test_5_other,
                            title = 'Feature Permutation Importance for Classifying next_type_other (Main Pool: 5)')

# Figure 6.6c
plot_permutation_importance(model_30_main, X_test_30_main, y_test_30_main,
                            title = 'Feature Permutation Importance for Classifying next_type_main (Main Pool: 30)')

# Figure 6.6d
plot_permutation_importance(model_30_other, X_test_30_other, y_test_30_other,
                            title = 'Feature Permutation Importance for Classifying next_type_other (Main Pool: 30)')


""" Dataset Shift Inspection """

# Calculates KL divergence from each feature's training/validation distribution to its test distribution
def kl_divergence_train_test(X_train_val, X_test):
    n_features = X_train_val.shape[1]
    kl_divergence = []

    for i in range(n_features):
        # Extract the i-th feature from both train/validation and test data
        train_feature = X_train_val.iloc[:, i]
        test_feature = X_test.iloc[:, i]

        # Perform Kernel Density Estimation (KDE) on both the train/validation and test feature
        kde_train = gaussian_kde(train_feature)
        kde_test = gaussian_kde(test_feature)

        # Define a common range for the KDE
        min_val = min(train_feature.min(), test_feature.min())
        max_val = max(train_feature.max(), test_feature.max())
        x = np.linspace(min_val, max_val, 1000) 

        # Calculate the PDFs of train/validation and test features
        pdf_train = kde_train(x)
        pdf_test = kde_test(x)

        # Calculate the KL divergence between the two PDFs
        kl_div = entropy(pdf_train, pdf_test)
        kl_divergence.append(kl_div)

    # Create a Pandas Series with feature names as index
    kl_divergence = pd.Series(kl_divergence, index=X_train_val.columns, name='KL Divergence')
    
    return kl_divergence

# Table 6.7 (USDC-WETH-0.0005 Dataset)
kl_divergence_train_test(X_train_val_30_main, X_test_30_main)

# Table 6.7 (USDC-WETH-0.003 Dataset)
kl_divergence_train_test(X_train_val_30_main, X_test_30_main)
