# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

# Display 4 decimal points
pd.options.display.float_format = '{:.4f}'.format

# Plot style
sns.set(style="darkgrid")

# Create the Pool 5 and Pool 30 DataFrames
usdc_weth_5 = pd.read_csv("usdc_weth_5_final.csv")
usdc_weth_30 = pd.read_csv("usdc_weth_30_final.csv")


""" Feature Analysis """

""" The USDC-WETH-0.0005 Dataset """

# Select the numerical features from DataFrames
numerical_features = usdc_weth_5.select_dtypes(include="float").drop(columns=["price"])

# Create a Pool 5 numerical feature matrix
X_5_numerical = usdc_weth_5[numerical_features.columns]

# Table 4.2
sumstats_5 = X_5_numerical[["s0", "w0", "b1_mint_main", "b3_burn_other", "depth_diff", "depth_ratio"]].describe()
print(sumstats_5)

# Plots a scatterplot of features_to_compare from the input DataFrame before and after applying StandardScaler
def plot_scaler_scales(data, features_to_compare, title):
    # Create a copy of the input DataFrame
    data_scaled = data.copy()

    scaler = StandardScaler()

    data_scaled[features_to_compare] = scaler.fit_transform(data_scaled[features_to_compare])

    # Create a 1x2 grid of scatterplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # First scatterplot for the raw features
    sns.scatterplot(data=data, x=features_to_compare[0], y=features_to_compare[1], ax=axes[0])
    axes[0].set_title("Raw Feature Scales", fontsize=18, y=1.02)
    axes[0].tick_params(axis='x', labelsize=12)  
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].set_xlabel(features_to_compare[0], fontsize=14) 
    axes[0].set_ylabel(features_to_compare[1], fontsize=14) 

    # Second scatterplot for the scaled features
    sns.scatterplot(data=data_scaled, x=features_to_compare[0], y=features_to_compare[1], ax=axes[1])
    axes[1].set_title("Standard Scaled Feature Scales", fontsize=18, y=1.02)
    axes[1].tick_params(axis='x', labelsize=12)  
    axes[1].tick_params(axis='y', labelsize=12) 
    axes[1].set_xlabel(features_to_compare[0], fontsize=14) 
    axes[1].set_ylabel(features_to_compare[1], fontsize=14) 

    plt.suptitle(title, fontsize=22)

    plt.tight_layout()
    plt.show()

# Figure 4.1
plot_scaler_scales(X_5_numerical, ["s0", "b1_mint_main"], "StandardScaler Feature Comparison for USDC-WETH-0.0005")

# Show numerical features for Pool 5 are skewed
X_5_numerical.apply(lambda x: x.skew())

# Plots a scatterplot and marginal distributions of features_to_compare from the input DataFrame after applying PowerTransformer
def plot_transformer_scales(data, features_to_compare, title):
    # Create a copy of the input DataFrame
    data_trans = data.copy()

    pt = PowerTransformer()

    data_trans[features_to_compare] = pt.fit_transform(data_trans[features_to_compare])

    # Create jointplot
    g = sns.jointplot(data=data_trans, x=features_to_compare[0], y=features_to_compare[1], kind="scatter", height=4, ratio=5)

    g.fig.dpi = 300

    # Add a title to the jointplot
    g.fig.suptitle(title, size=12, y=1.01)

    # Adjust the size of axis labels
    g.ax_joint.set_xlabel(features_to_compare[0], fontsize=10)
    g.ax_joint.set_ylabel(features_to_compare[1], fontsize=10)

    plt.show()

# Figure 4.2
plot_transformer_scales(X_5_numerical, ["s0", "b1_mint_main"], title="Power Transformed Feature Scales for USDC-WETH-0.0005")

# Plots a correlation heatmap of the input DataFrame
def plot_corr_mat(data, title):
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Create a heatmap 
    plt.figure(figsize=(12, 10)) 
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)
    plt.title(title, fontsize=20)
    plt.xticks(rotation=90, fontsize = 14)
    plt.yticks(rotation=0, fontsize = 14)
    plt.show()

# Figure 4.3
plot_corr_mat(X_5_numerical, title="Correlation Heatmap of USDC-WETH-0.0005 Numerical Features")


""" The USDC-WETH-0.003 Dataset """

# Create a Pool 30 numerical feature matrix
X_30_numerical = usdc_weth_30[numerical_features.columns]

# Table 4.3
sumstats_30 = X_30_numerical[["s0", "w0", "b1_mint_main", "b3_burn_other", "depth_diff", "depth_ratio"]].describe()
print(sumstats_30)

# Figure 4.4
plot_scaler_scales(X_30_numerical, ["w0", "depth_ratio"], "StandardScaler Feature Comparison for USDC-WETH-0.003")

# Show numerical features for Pool 30 are skewed
X_30_numerical.apply(lambda x: x.skew())

# Figure 4.5 
plot_transformer_scales(X_30_numerical, ["w0", "depth_ratio"], title="Power Transformed Feature Scales for USDC-WETH-0.003")

# Figure 4.6
plot_corr_mat(X_30_numerical, title="Correlation Heatmap of USDC-WETH-0.003 Numerical Features")


""" Target Analysis """

# Pool 5 classification targets
y_clf_5 = usdc_weth_5[["next_type_main", "next_type_other"]]

# Table 4.4
y_clf_5.describe()

# Pool 30 classification targets
y_clf_30 = usdc_weth_30[["next_type_main", "next_type_other"]]

# Table 4.4
y_clf_30.describe()

# Pool 5 regression targets
y_reg_5 = usdc_weth_5[["next_mint_time_main", "next_burn_time_main", "next_mint_time_other", "next_burn_time_other"]]

# Table 4.5
y_reg_5.describe()

# Pool 30 regression targets
y_reg_30 = usdc_weth_30[["next_mint_time_main", "next_burn_time_main", "next_mint_time_other", "next_burn_time_other"]]

# Table 4.6
y_reg_30.describe()

# Plots marginal distributions of input targets before and after a log1p transformation
def plot_reg_dists(targets, title):
    # Create a 4x2 grid of plots
    fig, axes = plt.subplots(4, 2, figsize=(8, 10), dpi = 300)

    fig.suptitle(title, fontsize=14)

    # Iterate through each column and create distribution plots
    for i, col in enumerate(y_reg_5.columns):
        # Plot the original distribution
        sns.histplot(data=targets, x=col, ax=axes[i, 0], kde=True, bins=40, color='red')

        # Apply log1p transformation and plot the transformed distribution
        transformed_col = np.log1p(targets[col])
        sns.histplot(data=transformed_col, ax=axes[i, 1], kde=True, bins=40, color='green')

        axes[i, 0].xaxis.label.set_fontsize(14)
        axes[i, 0].yaxis.label.set_fontsize(14)
        axes[i, 1].xaxis.label.set_fontsize(14)
        axes[i, 1].yaxis.label.set_fontsize(14)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) 
    plt.show()

# Figure 4.7
plot_reg_dists(y_reg_5, "USDC-WETH-0.0005 Target Distributions Before and After Log1p Transformation")

# Figure 4.8
plot_reg_dists(y_reg_30, "USDC-WETH-0.003 Target Distributions Before and After Log1p Transformation")