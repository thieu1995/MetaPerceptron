#!/usr/bin/env python
# Created by "Thieu" at 19:27, 16/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from metaperceptron import MhaMlpTuner


## Load data object
X, y = load_diabetes(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape)

# Example parameter grid of tuning hyper-parameter for Genetic Algorithm-based MLP
param_dict = {
    'hidden_layers': [30, 40],
    'act_names': ['Tanh', 'ELU'],
    'dropout_rates': [0.2, None],
    'optim': ['BaseGA'],
    'optim_params': [
        {"epoch": 10, "pop_size": 20},
        {"epoch": 20, "pop_size": 20},
    ],
    'obj_name': ["MSE"],
    'seed': [42],
    "verbose": [False],
}

# Initialize the tuner
tuner = MhaMlpTuner(
    task="regression",
    param_dict=param_dict,
    search_method="gridsearch",  # or "randomsearch"
    scoring='MSE',
    cv=3,
    verbose=2,          # Example additional argument
    n_jobs=4            # Parallelization
)

# Perform tuning
tuner.fit(X_train, y_train)
print("Best Parameters: ", tuner.best_params_)
print("Best Estimator: ", tuner.best_estimator_)

y_pred = tuner.predict(X_test)
print(tuner.best_estimator_.evaluate(y_test, y_pred, list_metrics=["MSE", "MAPE", "R2", "KGE", "NSE"]))
