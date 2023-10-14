# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import time
import random
import joblib
from scipy.stats import uniform, loguniform, randint, norm
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
#Classification methods 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Display 4 decimal points
pd.options.display.float_format = '{:.4f}'.format

# Define seed number to control randomness
rng = 2023


""" Classification Pipeline for next_type_main in Pool 5 """

# Create the Pool 5 DataFrame
usdc_weth_5 = pd.read_csv("usdc_weth_5_final.csv")

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

# Create our StandardScaler and PowerTransformer
custom_std_scaler = CustomStandardScaler()
custom_pow_transformer = CustomPowerTransformer()

# Creates an XGBClassifier that is well-behaved for our target format
class EncodedXGBClassifier(XGBClassifier):
    encoding = {"burn": 0, "mint": 1}
    decoding = np.array(["burn", "mint"])

    def _init_(self, *args, **kwargs):
        super()._init_(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        y_enc = y.map(EncodedXGBClassifier.encoding)
        super().fit(X, y_enc, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        prediction_enc = super().predict(X, *args, **kwargs)
        prediction_dec = EncodedXGBClassifier.decoding[prediction_enc]
        return prediction_dec

    def score(self, X, y, *args, **kwargs):
        prediction = self.predict(X)
        return np.mean(prediction == y)

# Applies RandomizedSearchCV with TimeSeriesSplit cross-validation for the input classifier
def fit_model_clf(X_train_val, y_train_val, clf, clf_param_grid, tree = False, rng=rng):
    
    pipe = Pipeline(steps=[("scaler", None), ("clf", clf)])
    
    # Only add feature transformations for non-tree-based methods
    if not tree:
        param_grid = {
            'scaler': [custom_std_scaler, custom_pow_transformer]
        }
        param_grid.update(clf_param_grid)
    else:
        param_grid = clf_param_grid
    
    # Create a RandomizedSearchCV object with TimeSeriesSplit cross-validation
    grid = RandomizedSearchCV(
        pipe, param_distributions=param_grid, n_iter=100,
        scoring='accuracy', refit=True, cv=TimeSeriesSplit(n_splits=5).split(X_train_val),
        random_state = rng,
        n_jobs = 4 # Should be customized for the user's computer
    )

    # Conduct RandomizedSearchCV for the input classifier on the training/validation split
    grid.fit(X_train_val, y_train_val)
    
    print("Classifier: ", clf.__class__.__name__)
    print("Best params: ", grid.best_params_)
    print("Accuracy: ", np.round(grid.best_score_, 4))

    return grid

# Define all classification methods and their hyperparameter distributions
# Linear classifiers
logit = LogisticRegression(penalty = 'l2')

logit_param_grid = {
    'clf__C': loguniform(1e-4, 1),
    'clf__max_iter': randint(100, 1000),
}

ridge_clf = RidgeClassifier()

ridge_clf_param_grid = {
    'clf__alpha': loguniform(1e-1, 1e2),
}

# Distance-based classifiers
svc = SVC()

svc_param_grid = {
    'clf__kernel': ["rbf", "poly"],
    'clf__C': loguniform(1e-4, 1e3),
    'clf__gamma': loguniform(1e-6, 1e2)
}

knnc = KNeighborsClassifier()

knnc_param_grid = {
    'clf__n_neighbors': randint(1, 100),
    'clf__weights': ["uniform", "distance"],
    'clf__p': [1, 2]
}

# Neural networks
mlpc = MLPClassifier(max_iter = 1000, shuffle = False, random_state = rng)

mlpc_param_grid = {
    'clf__hidden_layer_sizes': [(100,), (200,), (100, 50), (100, 100), (200, 100)],
    'clf__alpha': loguniform(1e1, 1e6)
}

# Tree-based methods
dtreec = DecisionTreeClassifier(random_state=rng)

dtreec_param_grid = {
    'clf__max_depth': [None] + [random.randint(1, 20) for _ in range(10)],
    'clf__min_samples_leaf': randint(3, 20),
}

rfc = RandomForestClassifier(random_state=rng, max_samples = 0.75)

rfc_param_grid = {
    'clf__n_estimators': randint(50, 250),
    'clf__max_features': ["sqrt", "log2", None]
}

adac = AdaBoostClassifier(random_state=rng)

adac_param_grid = {
    'clf__n_estimators': randint(30, 150),
    # Use estimator instead of base_estimator if sklearn > version 1.4
    #'clf_estimator': [None, DecisionTreeClassifier(max_depth=3)]
    'clf__base_estimator': [None, DecisionTreeClassifier(max_depth = 3)]
}

# Gradient boosting 
xgbc = EncodedXGBClassifier(booster='gbtree', subsample = 0.75, 
                            colsample_bylevel = np.sqrt(46)/46, random_state = rng)

xgbc_param_grid = {
    'clf__n_estimators': randint(100, 200),
    'clf__learning_rate': loguniform(1e-4, 1e-1),
    'clf__min_split_loss': uniform(0.1, 1.9),
    'clf__max_depth': randint(1, 6),
    'clf__reg_alpha': loguniform(1e-1, 1e3),
    'clf__reg_lambda': loguniform(1e-1, 1e3)
}

lgbmc = LGBMClassifier(boosting_type='goss', objective = 'binary', 
                       top_rate = 0.7, other_rate = 0.3, verbose = -1, random_state = rng)

lgbmc_param_grid = {
    'clf__n_estimators': randint(100, 200),
    'clf__num_leaves': randint(2, 32),
    'clf__learning_rate': loguniform(1e-4, 1e-1),
    'clf__feature_fraction_bynode': [np.sqrt(46)/46, math.log2(46)/46, 0.25, 1],
    'clf__max_bin': [255, 510] + [random.randint(255, 510) for _ in range(10)],
    'clf__reg_alpha': loguniform(1e-1, 1e3),
    'clf__reg_lambda': loguniform(1e-1, 1e3)
    
}

catc = CatBoostClassifier(boosting_type = "Ordered", random_seed = rng, verbose = False)

catc_param_grid = {
    'clf__n_estimators': randint(100, 200),
    'clf__learning_rate': loguniform(1e-4, 1e-1),
    'clf__depth': randint(1, 6),
    'clf__leaf_estimation_iterations': [1, 5],
    'clf__rsm': [np.sqrt(46)/46, math.log2(46)/46, 1],
    'clf__reg_lambda': loguniform(1e-1, 1e3)
}

# Group methods and their parameter distributions for the pipeline functions
non_tree_models_clf = [
    (logit, logit_param_grid),
    (ridge_clf, ridge_clf_param_grid),
    (svc, svc_param_grid),
    (knnc, knnc_param_grid),
    (mlpc, mlpc_param_grid)
]
tree_models_clf = [
    (dtreec, dtreec_param_grid),
    (rfc, rfc_param_grid),
    (adac, adac_param_grid),
    (xgbc, xgbc_param_grid),
    (lgbmc, lgbmc_param_grid),
    (catc, catc_param_grid)
]

# Applies fit_model_clf for all methods and creates a DataFrame with cross-validation results
def individual_training_clf(X_train_val, y_train_val):
    individual_training_results = []
    
    # Iterate through all non-tree-based methods
    for model, param_grid in non_tree_models_clf:
        start_time = time.time()
        grid = fit_model_clf(X_train_val, y_train_val, model, param_grid, tree=False)
        end_time = time.time()
        train_time = end_time - start_time
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_
        best_score = np.round(grid.best_score_, 4)
        
        individual_training_results.append({
            'Classifier': model.__class__.__name__,
            'Randomized Search (s)': train_time,
            'Best Parameters': best_params,
            'Best Estimator': best_estimator,
            'Accuracy': best_score
        })
    
    # Iterate through all tree-based methods
    for model, param_grid in tree_models_clf:
        start_time = time.time()
        grid = fit_model_clf(X_train_val, y_train_val, model, param_grid, tree=True)
        end_time = time.time()
        train_time = end_time - start_time
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_
        best_score = np.round(grid.best_score_, 4)
        
        individual_training_results.append({
            'Classifier': model.__class__.__name__,
            'Randomized Search (s)': train_time,
            'Best Parameters': best_params,
            'Best Estimator': best_estimator,
            'Accuracy': best_score
        })
    
    individual_training_results_df = pd.DataFrame(individual_training_results)
    return individual_training_results_df

# Inputs the DataFrame from individual_training_clf to predict on the test split with each best estimator
def prediction_results_clf(X_test, y_test, X_train_val, y_train_val, training_results):
    prediction_results_list = []

    # Iterate through all methods
    for index, row in training_results.iterrows():
        model_name = row['Classifier']
        best_estimator = row['Best Estimator']
        best_params = row['Best Parameters']

        # Make predictions with the best estimator on the test split
        start_time = time.time() 
        y_pred = best_estimator.predict(X_test)
        end_time = time.time() 

        prediction_time = end_time - start_time 

        acc = accuracy_score(y_test, y_pred) 
        
        # 95% Wald test split accuracy confidence interval
        alpha = 0.05
        z = norm.ppf(q=(alpha/2, 1-alpha/2))
        var_hat = acc*(1-acc)/len(y_test)
        
        # Make predictions with the best estimator on the training/validation split
        y_train_val_pred = best_estimator.predict(X_train_val)
        acc_train = accuracy_score(y_train_val, y_train_val_pred)

        prediction_results_list.append({
            'Classifier': model_name,
            'Test Accuracy': acc,
            '95% C.I.': np.round(acc + z * np.sqrt(var_hat), 4),
            'Train Accuracy': acc_train,
            'Randomized Search (s)': row['Randomized Search (s)'],
            'Prediction (s)': prediction_time,
            'Best Parameters': best_params
        })
        
    prediction_results_df = pd.DataFrame(prediction_results_list)
    return prediction_results_df

# Create training/validation and test splits for targeting next_type_main in Pool 5
X_train_val_5_main, X_test_5_main, y_train_val_5_main, y_test_5_main = process_dataframe_clf(usdc_weth_5, "next_type_main")

# Apply fit_model_clf for all methods and creates a DataFrame with cross-validation results
trained_models_5_main = individual_training_clf(X_train_val_5_main, y_train_val_5_main)

# Inspect cross-validation results
trained_models_5_main

# Make predictions on the test split with each best estimator 
test_pred_5_main = prediction_results_clf(X_test_5_main, y_test_5_main, 
                                          X_train_val_5_main, y_train_val_5_main, 
                                          trained_models_5_main)

# Table B.1
test_pred_5_main

# Save the best estimator (RidgeClassifier)
joblib.dump(trained_models_5_main.loc[1, 'Best Estimator'], "ridge_5_main_final.joblib")