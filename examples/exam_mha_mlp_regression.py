#!/usr/bin/env python
# Created by "Thieu" at 21:35, 02/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.datasets import load_diabetes
from metaperceptron import Data, MhaMlpRegressor


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("minmax", ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

## Create model
model = MhaMlpRegressor(hidden_layers=(30, 15,), act_names="ELU", dropout_rates=0.2, act_output=None,
                        optim="BaseGA", optim_params={"name": "WOA", "epoch": 10, "pop_size": 30},
                        obj_name="MSE", seed=42, verbose=True,
                        lb=None, ub=None, mode='single', n_workers=None, termination=None)
## Train the model
model.fit(data.X_train, data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))

print(model.optimizer.name)
