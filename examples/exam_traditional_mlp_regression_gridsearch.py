#!/usr/bin/env python
# Created by "Thieu" at 09:18, 17/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metaperceptron import Data, MlpRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

## Set up parameters for MLP
params = {
    "hidden_size": [10, 20],
    "act1_name": ["elu", "tanh"], #  ['relu', 'sigmoid', 'hardsigmoid', 'tanh', 'elu', 'celu', 'selu', 'glu', 'gelu'],
    "act2_name": ["elu", "tanh"], #  ['relu', 'sigmoid', 'hardsigmoid', 'tanh', 'elu', 'celu', 'selu', 'glu', 'gelu'],
    "obj_name": ["MAE", "MSE"],
    "max_epochs": [10, 20],
    "batch_size": [16],
    "optimizer": ["Adam"], # ["Adadelta", "Adam", "Adamax", "RMSprop", "SGD"],
    "optimizer_paras": [
        {"lr": 0.01,},
        {"lr": 0.02,},
    ]
}

## Define the core
model = MlpRegressor(verbose=False)

## Define the gridsearch object
gs = GridSearchCV(model, params, refit=True, cv=3, verbose=2)

## Train the gridsearch
gs.fit(data.X_train, data.y_train)

## Get the best score and best parameter
print(gs.best_score_, gs.best_params_)

# Get the core with the best parameters
best_model = gs.best_estimator_

## Get the prediction of the best core
y_pred = best_model.predict(data.X_test)

## Calculate some metrics
print(best_model.score(X=data.X_test, y=data.y_test, method="RMSE"))
print(best_model.scores(X=data.X_test, y=data.y_test, list_methods=["R2", "NSE", "MAPE"]))
print(best_model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))
