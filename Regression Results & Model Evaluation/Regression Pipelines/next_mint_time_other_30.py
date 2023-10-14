# Required libraries
import pandas as pd
import numpy as np
import math
import time
import random
import joblib
from scipy.stats import uniform, loguniform, randint, norm
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, train_test_split, RandomizedSearchCV
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

# Display 4 decimal points
pd.options.display.float_format = '{:.4f}'.format

# Define seed number to control randomness
rng = 2023

""" Regression Pipeline for next_mint_time_other in Pool 30"""

# Create the Pool 5 DataFrame
usdc_weth_30 = pd.read_csv("usdc_weth_30_final.csv")

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

# Create our StandardScaler and PowerTransformer
custom_std_scaler = CustomStandardScaler()
custom_pow_transformer = CustomPowerTransformer()

# Applies RandomizedSearchCV with TimeSeriesSplit cross-validation for the input regressor
def fit_model_regr(X_train_val, y_train_val, regr, regr_param_grid, tree = False, rng=rng):
    
    pipe = Pipeline(steps=[("scaler", None), ("regr", regr)])
    
    # Only add feature transformations for non-tree-based methods
    if not tree:
        param_grid = {
            'scaler': [custom_std_scaler, custom_pow_transformer]
        }
        param_grid.update(regr_param_grid)
    else:
        param_grid = regr_param_grid
        

    # Create a RandomizedSearchCV object with TimeSeriesSplit cross-validation
    grid = RandomizedSearchCV(
        pipe, param_distributions=param_grid, n_iter=100,
        scoring='neg_mean_squared_error', refit=True, cv=TimeSeriesSplit(n_splits=5).split(X_train_val),
        random_state = rng, 
        n_jobs = 2 # Should be customized for the user's computer
    )

    # Conduct RandomizedSearchCV for the input regressor on the training/validation split
    grid.fit(X_train_val, y_train_val)
    
    print("Regressor: ", regr.__class__.__name__)
    print("Best params: ", grid.best_params_)
    print("Neg MSLE: ", np.round(grid.best_score_, 4))

    return grid

# Define all regression methods and their hyperparameter distributions
# Linear regressors
ridge = Ridge()

ridge_param_grid = {
    'regr__alpha': loguniform(1e-1, 1e4)
}

lasso = Lasso()

lasso_param_grid = {
    'regr__alpha': loguniform(1e-1, 1e4)
}

enet = ElasticNet()

enet_param_grid = {
    'regr__alpha': loguniform(1e-1, 1e4),
    'regr__l1_ratio': uniform(0.05, 0.9)
}

# Distance-based regressors
krr = KernelRidge()

krr_param_grid = {
    'regr__alpha': loguniform(1e-1, 1e4),
    'regr__kernel': ["rbf", "poly"],
    'regr__gamma': loguniform(1e-6, 1e2)
}

svr = SVR()

svr_param_grid = {
    'regr__kernel': ["rbf", "poly"],
    'regr__C': loguniform(1e-4, 1e3),
    'regr__gamma': loguniform(1e-6, 1e2),
    'regr__epsilon': uniform(0.1, 1.4)
}

knn = KNeighborsRegressor()

knn_param_grid = {
    'regr__n_neighbors': randint(1, 100),
    'regr__weights': ["uniform", "distance"],
    'regr__p': [1, 2]
}

# Neural Networks
mlpr = MLPRegressor(max_iter = 500, shuffle = False, random_state = rng)

mlpr_param_grid = {
    'regr__hidden_layer_sizes': [(100,), (200,), (100, 50), (100, 100), (200, 100)],
    'regr__alpha': loguniform(1e1, 1e6)
}

# Tree-based methods
dtree = DecisionTreeRegressor(random_state=rng)

dtree_param_grid = {
    'regr__max_depth': [None] + [random.randint(1, 20) for _ in range(10)],
    'regr__min_samples_leaf': randint(3, 20)
}

rf = RandomForestRegressor(random_state=rng, max_samples = 0.75)

rf_param_grid = {
    'regr__n_estimators': randint(50, 250),
    'regr__max_features': ["sqrt", "log2", None]
}

ada = AdaBoostRegressor(loss='square', random_state=rng)

ada_param_grid = {
    'regr__n_estimators': randint(30, 150),
    # Use estimator instead of base_estimator if sklearn > version 1.4
    #'regr__estimator': [None, DecisionTreeRegressor(max_depth = 5)],
    'regr__base_estimator': [None, DecisionTreeRegressor(max_depth = 5)]
}

# Gradient boosting
xgb = XGBRegressor(booster='gbtree', subsample = 0.75, 
                            colsample_bylevel = np.sqrt(46)/46, random_state = rng)

xgb_param_grid = {
    'regr__n_estimators': randint(100, 200),
    'regr__learning_rate': loguniform(1e-4, 1e-1),
    'regr__min_split_loss': uniform(0.1, 1.9),
    'regr__max_depth': randint(1, 6),
    'regr__reg_alpha': loguniform(1e-1, 1e3),
    'regr__reg_lambda': loguniform(1e-1, 1e3)
}

lgbm = LGBMRegressor(boosting_type='goss', top_rate = 0.7, 
                     other_rate = 0.3, verbose = -1, random_state = rng)

lgbm_param_grid = {
    'regr__n_estimators': randint(100, 200),
    'regr__num_leaves': randint(2, 32),
    'regr__learning_rate': loguniform(1e-4, 1e-1),
    'regr__feature_fraction_bynode': [np.sqrt(46)/46, math.log2(46)/46, 0.25, 1],
    'regr__max_bin': [255, 510] + [random.randint(255, 510) for _ in range(10)],
    'regr__reg_alpha': loguniform(1e-1, 1e3),
    'regr__reg_lambda': loguniform(1e-1, 1e3)
}

cat = CatBoostRegressor(boosting_type = "Ordered", random_seed = rng, verbose = False)

cat_param_grid = {
    'regr__n_estimators': randint(100, 200),
    'regr__learning_rate': loguniform(1e-4, 1e-1),
    'regr__depth': randint(1, 6),
    'regr__leaf_estimation_iterations': [1, 5],
    'regr__rsm': [np.sqrt(46)/46, math.log2(46)/46, 1],
    'regr__reg_lambda': loguniform(1e-1, 1e3)
}

# Group methods and their parameter distributions for the pipeline functions
non_tree_models = [
    (ridge, ridge_param_grid),
    (lasso, lasso_param_grid),
    (enet, enet_param_grid),
    (krr, krr_param_grid),
    (svr, svr_param_grid),
    (knn, knn_param_grid),
    (mlpr, mlpr_param_grid)
]

tree_models = [
    (dtree, dtree_param_grid),
    (rf, rf_param_grid),
    (ada, ada_param_grid),
    (xgb, xgb_param_grid),
    (lgbm, lgbm_param_grid),
    (cat, cat_param_grid)
]

# Applies fit_model_regr for all methods and creates a DataFrame with cross-validation results
def individual_training_regr(X_train_val, y_train_val):
    individual_training_results = []
    
    # Iterate through all non-tree-based methods
    for model, param_grid in non_tree_models:
        start_time = time.time()
        grid = fit_model_regr(X_train_val, y_train_val, model, param_grid, tree=False)
        end_time = time.time()
        train_time = end_time - start_time
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_
        best_score = np.round(grid.best_score_, 4)
        
        individual_training_results.append({
            'Regressor': model.__class__.__name__,
            'Randomized Search (s)': train_time,
            'Best Parameters': best_params,
            'Best Estimator': best_estimator,
            'MSLE': best_score
        })
    
    # Iterate through all tree-based methods
    for model, param_grid in tree_models:
        start_time = time.time()
        grid = fit_model_regr(X_train_val, y_train_val, model, param_grid, tree=True)
        end_time = time.time()
        train_time = end_time - start_time
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_
        best_score = np.round(grid.best_score_, 4)
        
        individual_training_results.append({
            'Regressor': model.__class__.__name__,
            'Randomized Search (s)': train_time,
            'Best Parameters': best_params,
            'Best Estimator': best_estimator,
            'MSLE': best_score
        })
    
    individual_training_results_df = pd.DataFrame(individual_training_results)
    return individual_training_results_df

# Inputs the DataFrame from individual_training_regr to predict on the test split with each best estimator
def prediction_results_regr(X_test, y_test, X_train_val, y_train_val, training_results):
    prediction_results_list = []

    # Iterate through all methods
    for indeX_test, row in training_results.iterrows():
        model_name = row['Regressor']
        best_estimator = row['Best Estimator']
        best_params = row['Best Parameters']

        # Make predictions with the best estimator on the test split
        start_time = time.time() 
        y_pred = best_estimator.predict(X_test)
        end_time = time.time() 

        prediction_time = end_time - start_time 

        mse = mean_squared_error(y_test, y_pred) 
        
        # 95% Wald test split MSLE confidence interval
        alpha = 0.05
        z = norm.ppf(q=(alpha/2, 1-alpha/2))
        var_hat = np.var((y_pred - y_test)**2, ddof=1)/len(y_test)
        
        # Make predictions with the best estimator on the training/validation split
        y_train_val_pred = best_estimator.predict(X_train_val)
        mse_train = mean_squared_error(y_train_val, y_train_val_pred)

        prediction_results_list.append({
            'Regressor': model_name,
            'Test MSLE': mse,
            '95% C.I.': np.round(mse + z * np.sqrt(var_hat), 4),
            'Train MSLE': mse_train,
            'Randomized Search (s)': row['Randomized Search (s)'],
            'Prediction (s)': prediction_time,
            'Best Parameters': best_params
        })
        
    prediction_results_df = pd.DataFrame(prediction_results_list)
    return prediction_results_df

# Create training/validation and test splits for targeting next_mint_time_other in Pool 30
X_train_val_30_mint_other, X_test_30_mint_other, y_train_val_30_mint_other, y_test_30_mint_other = \
        process_dataframe_regr(usdc_weth_30, "next_mint_time_other")

# Apply fit_model_regr for all methods and create a DataFrame with cross-validation results
trained_models_30_mint_other = individual_training_regr(X_train_val_30_mint_other, y_train_val_30_mint_other)

# Inspect cross-validation results
trained_models_30_mint_other

# Make predictions on the test split with each best estimator
test_pred_30_mint_other = prediction_results_regr(X_test_30_mint_other, y_test_30_mint_other, 
                                                  X_train_val_30_mint_other, y_train_val_30_mint_other, 
                                                  trained_models_30_mint_other)

# Table C.7
test_pred_30_mint_other

# Save the best estimator (Ridge)
joblib.dump(trained_models_30_mint_other.loc[0, 'Best Estimator'], "ridge_30_mint_other_final.joblib")
