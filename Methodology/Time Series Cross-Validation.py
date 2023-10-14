# Required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

# Plot style 
sns.set(style="darkgrid")

# Plots a TimeSeriesSplit cross-validation toy example with 5 folds
def plot_ts_cv():
    # Generate sample time series data
    np.random.seed(2023)
    time_series = np.cumsum(np.random.randn(100))

    # Create a TimeSeriesSplit object with 5 folds
    tscv = TimeSeriesSplit(n_splits=5)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)

    # Initialize a counter for cross-validation folds
    cv_fold = 1

    legend_handles = []
    legend_labels = []

    for train_idx, test_idx in tscv.split(time_series):
        # Plot training data in blue
        train_fill = ax.fill_betweenx([cv_fold - 0.4, cv_fold + 0.4], train_idx[0], train_idx[-1], color=sns.color_palette()[0])

        # Plot test data in orange
        test_fill = ax.fill_betweenx([cv_fold - 0.4, cv_fold + 0.4], test_idx[0], test_idx[-1], color=sns.color_palette()[1])

        # Add handles and labels for legend
        if cv_fold == 1:
            legend_handles.extend([train_fill, test_fill])
            legend_labels.extend(['Training Split', 'Validation Split'])

        cv_fold += 1

    ax.set_xlabel('Time Index', fontsize = 16)
    ax.set_ylabel('Cross-Validation Fold', fontsize = 16)

    ax.tick_params(axis='both', labelsize=14)

    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title('TimeSeriesSplit Toy Example with 5 Cross-Validation Folds', fontsize = 20)

    ax.legend(legend_handles, legend_labels, loc='lower right', fontsize = 16)

    plt.tight_layout()
    plt.show()

# Figure 5.1
plot_ts_cv()
