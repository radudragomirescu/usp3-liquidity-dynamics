# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
# Regression methods
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge 
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# Regression model evaluation
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from sklearn.metrics import PredictionErrorDisplay
from sklearn.inspection import permutation_importance


""" Regression Model Evaluation Setup (Pool 30) """

# Create the Pool 30 DataFrame
usdc_weth_30 = pd.read_csv("usdc_weth_30_final.csv")

# Define seed number to control randomness
rng=2023

# Generates training/validation and test data splits for regression tasks
def process_dataframe_regr(dataframe, y_column):
    X = dataframe.drop(columns=['date', 'block_number', 'pool_name', 'price', 'next_type_main',
                                'next_type_other', 'next_mint_time_main', 'next_burn_time_main',
                                'next_mint_time_other', 'next_burn_time_other'])
    # One-hot encode the categorical features
    X = pd.get_dummies(X, prefix = ['type', 'liquidity'], drop_first = True)
    
    y = dataframe[y_column]
    # Log1p-transform the target
    y = np.log1p(y)
    
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

# Create training/validation and test splits for all regression targets in Pool 30
X_train_val_30_mint_main, X_test_30_mint_main, y_train_val_30_mint_main, y_test_30_mint_main = process_dataframe_regr(usdc_weth_30, "next_mint_time_main")
X_train_val_30_burn_main, X_test_30_burn_main, y_train_val_30_burn_main, y_test_30_burn_main = process_dataframe_regr(usdc_weth_30, "next_burn_time_main")
X_train_val_30_mint_other, X_test_30_mint_other, y_train_val_30_mint_other, y_test_30_mint_other = process_dataframe_regr(usdc_weth_30, "next_mint_time_other")
X_train_val_30_burn_other, X_test_30_burn_other, y_train_val_30_burn_other, y_test_30_burn_other = process_dataframe_regr(usdc_weth_30, "next_burn_time_other")

# Load in the best estimator for each regression target in Pool 30 
model_30_mint_main = joblib.load("xgb_30_mint_main_final.joblib")
model_30_burn_main = joblib.load("rf_30_burn_main_final.joblib")
model_30_mint_other = joblib.load("ridge_30_mint_other_final.joblib")
model_30_burn_other = joblib.load("knn_30_burn_other_final.joblib")

# Make predictions on the test split of each target with the respective best estimator
y_pred_30_mint_main = model_30_mint_main.predict(X_test_30_mint_main)
y_pred_30_burn_main = model_30_burn_main.predict(X_test_30_burn_main)
y_pred_30_mint_other = model_30_mint_other.predict(X_test_30_mint_other)
y_pred_30_burn_other = model_30_burn_other.predict(X_test_30_burn_other)

# Make predictions on the training/validation split of each target with the respective best estimator
y_train_pred_30_mint_main = model_30_mint_main.predict(X_train_val_30_mint_main)
y_train_pred_30_burn_main = model_30_burn_main.predict(X_train_val_30_burn_main)
y_train_pred_30_mint_other = model_30_mint_other.predict(X_train_val_30_mint_other)
y_train_pred_30_burn_other = model_30_burn_other.predict(X_train_val_30_burn_other)

# Make predictions on the test split with the baseline persistence model for each target
dummy_y_pred_30_mint_main = np.log1p(X_test_30_mint_main['b1_mint_main'])
dummy_y_pred_30_burn_main = np.log1p(X_test_30_burn_main['b1_burn_main'])
dummy_y_pred_30_mint_other = np.log1p(X_test_30_mint_other['b1_mint_other'])
dummy_y_pred_30_burn_other = np.log1p(X_test_30_burn_other['b1_burn_other'])

# Plot style
sns.set_style("darkgrid")


""" Regression Model Evaluation and Plots (Pool 30) """

# Table 7.3 (Persistence Accuracy)
print(mean_squared_error(y_test_30_mint_main, dummy_y_pred_30_mint_main))
print(mean_squared_error(y_test_30_burn_main, dummy_y_pred_30_burn_main))
print(mean_squared_error(y_test_30_mint_other, dummy_y_pred_30_mint_other))
print(mean_squared_error(y_test_30_burn_other, dummy_y_pred_30_burn_other))

# Table 7.3 (Best Estimator Accuracy) 
print(mean_squared_error(y_test_30_mint_main, y_pred_30_mint_main))
print(mean_squared_error(y_test_30_burn_main, y_pred_30_burn_main))
print(mean_squared_error(y_test_30_mint_other, y_pred_30_mint_other))
print(mean_squared_error(y_test_30_burn_other, y_pred_30_burn_other))

# Plots training/validation and test marginal distributions for the input target and predictions
def plot_marginal_dists(y_train, y_train_pred, y_test, y_pred, title="Marginal Distributions of Target"):
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi = 300)

    # Plot histograms for y_train and y_train_pred on the left side
    sns.histplot(y_train, ax=axes[0, 0], kde=True, color='green', label='y_train')
    sns.histplot(y_train_pred, ax=axes[1, 0], kde=True, color='red', label='y_train_pred')

    # Plot histograms for y_test and y_pred on the right side
    sns.histplot(y_test, ax=axes[0, 1], kde=True, color='green', label='y_test')
    sns.histplot(y_pred, ax=axes[1, 1], kde=True, color='red', label='y_test_pred')

    # Subplot titles
    axes[0, 0].set_title('Training/Validation Split Target Distributions', fontsize=18, y= 1.02)
    axes[0, 1].set_title('Test Split Target Distributions', fontsize=18, y=1.02)

    # Add legends
    axes[0, 0].legend(fontsize=12)
    axes[1, 0].legend(fontsize=12)
    axes[0, 1].legend(fontsize=12)
    axes[1, 1].legend(fontsize=12)
    
    # Set x-labels and y-labels
    for ax in axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        
        ax.tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    
    # Main title
    fig.suptitle(title, fontsize=26, y = 1.06)

    plt.show()

# Figure 7.9
plot_marginal_dists(y_train_val_30_mint_main, y_train_pred_30_mint_main, y_test_30_mint_main, y_pred_30_mint_main,
                   title="Marginal Distributions of next_mint_time_main (Main Pool: 30)")

# Figure 7.11
plot_marginal_dists(y_train_val_30_burn_main, y_train_pred_30_burn_main, y_test_30_burn_main, y_pred_30_burn_main,
                   title="Marginal Distributions of next_burn_time_main (Main Pool: 30)")

# Figure 7.13
plot_marginal_dists(y_train_val_30_mint_other, y_train_pred_30_mint_other, y_test_30_mint_other, y_pred_30_mint_other,
                   title="Marginal Distributions of next_mint_time_other (Main Pool: 30)")

# Figure 7.15
plot_marginal_dists(y_train_val_30_burn_other, y_train_pred_30_burn_other, y_test_30_burn_other, y_pred_30_burn_other,
                   title="Marginal Distributions of next_burn_time_other (Main Pool: 30)")

# Helper function to generate KDEs
def kde_normalize_array(arr):
    kde = gaussian_kde(arr)
    x = np.linspace(min(arr), max(arr), 1000) 
    normalized_arr = kde(x)
    normalized_arr /= np.sum(normalized_arr) 
    return normalized_arr

# Computes KL divergence between input predictions and target for training/validation and test splits
def calculate_kl_divergence(y_train, y_train_pred, y_test, y_test_pred):
    # Normalize arrays into probability distributions
    n_y_train = kde_normalize_array(y_train)
    n_y_train_pred = kde_normalize_array(y_train_pred)
    n_y_test = kde_normalize_array(y_test)
    n_y_test_pred = kde_normalize_array(y_test_pred)
    
    print("KL Divergence - Train/Validation:", np.round(entropy(n_y_train, n_y_train_pred), 4))
    print("KL Divergence - Test:", np.round(entropy(n_y_test, n_y_test_pred), 4))

# Table 7.4 (first row)
calculate_kl_divergence(y_train_val_30_mint_main, y_train_pred_30_mint_main, y_test_30_mint_main, y_pred_30_mint_main)

# Table 7.4 (second row)
calculate_kl_divergence(y_train_val_30_burn_main, y_train_pred_30_burn_main, y_test_30_burn_main, y_pred_30_burn_main)

# Table 7.4 (third row)
calculate_kl_divergence(y_train_val_30_mint_other, y_train_pred_30_mint_other, y_test_30_mint_other, y_pred_30_mint_other)

# Table 7.4 (fourth row)
calculate_kl_divergence(y_train_val_30_burn_other, y_train_pred_30_burn_other, y_test_30_burn_other, y_pred_30_burn_other)

# Plots actual vs. predicted and residuals vs. predicted plots for the input test split target and predictions
def plot_prediction_error(y_test, y_pred, title = "Regression Performance Evaluation"):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi = 300)
    
    # Plot Actual vs. Predicted values
    PredictionErrorDisplay.from_predictions(y_test, y_pred, kind="actual_vs_predicted", ax=axs[0])
    axs[0].set_title("Test Split Actual vs. Predicted Values", fontsize=18)
    axs[0].set_xlabel("Predicted Values", fontsize=14)
    axs[0].set_ylabel("Actual Values", fontsize=14)
    axs[0].tick_params(axis='both', labelsize=14)
    
    # Plot Residuals vs. Predicted Values
    PredictionErrorDisplay.from_predictions(y_test, y_pred, kind="residual_vs_predicted", ax=axs[1])
    axs[1].set_title("Test Split Residuals vs. Predicted Values", fontsize=18)
    axs[1].set_xlabel("Predicted Values", fontsize=14)
    axs[1].set_ylabel("Residuals", fontsize=14)
    axs[1].tick_params(axis='both', labelsize=14)
    
    fig.suptitle(title, fontsize=22, y = 1.01)
    plt.tight_layout()
    plt.show()

# Figure 7.10
plot_prediction_error(y_test_30_mint_main, y_pred_30_mint_main, 
                     title="Prediction Error Analysis for next_mint_time_main (Main Pool: 30)")

# Figure 7.12
plot_prediction_error(y_test_30_burn_main, y_pred_30_burn_main, 
                     title="Prediction Error Analysis for next_burn_time_main (Main Pool: 30)")

# Figure 7.14
plot_prediction_error(y_test_30_mint_other, y_pred_30_mint_other, 
                     title="Prediction Error Analysis for next_mint_time_other (Main Pool: 30)")

# Figure 7.16
plot_prediction_error(y_test_30_burn_other, y_pred_30_burn_other, 
                     title="Prediction Error Analysis for next_burn_time_other (Main Pool: 30)")

# Plots top 10 features by mean feature importance over 50 iterations on the test split for the input regressor
def plot_permutation_importance(model, X_test, y_test, title="Permutation Importance", top_n=10, n_repeats=50, rng=rng):
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, scoring='neg_mean_squared_error', 
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
    plt.figure(figsize=(8, 5), dpi = 300)
    ax = sns.barplot(data=importance_df, x='Mean_Importance', y='Feature', palette='viridis')
    for i, (_, row) in enumerate(importance_df.iterrows()):
        ax.text(row['Mean_Importance'], i, f" ± {row['Std_Importance']:.4f}", va='center', ha='left', color='black')
    plt.title(title, y = 1.02, fontsize = 16)
    plt.xlabel('Mean Increase in MSLE (± SD)', fontsize = 12)
    plt.ylabel('')  # Remove the y-label
    plt.xlim(right=max(importance_df['Mean_Importance']) + 0.008) 
    
    ax.tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    plt.show()

# Figure 7.18a
plot_permutation_importance(model_30_mint_main, X_test_30_mint_main, y_test_30_mint_main,
                            title = 'Feature Permutation Importance for Predicting next_mint_time_main (Main Pool: 30)')

# Figure 7.18b
plot_permutation_importance(model_30_burn_main, X_test_30_burn_main, y_test_30_burn_main,
                            title = 'Feature Permutation Importance for Predicting next_burn_time_main (Main Pool: 30)')

# Figure 7.18c
plot_permutation_importance(model_30_mint_other, X_test_30_mint_other, y_test_30_mint_other,
                            title = 'Feature Permutation Importance for Predicting next_mint_time_other (Main Pool: 30)')

# Figure 7.18d
plot_permutation_importance(model_30_burn_other, X_test_30_burn_other, y_test_30_burn_other,
                            title = 'Feature Permutation Importance for Predicting next_burn_time_other (Main Pool: 30)')