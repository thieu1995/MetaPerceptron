#!/usr/bin/env python
# Created by "Thieu" at 21:35, 02/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_iris
from metaperceptron import Data, MhaMlpClassifier


## Load data object
X, y = load_iris(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True, shuffle=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

## Create model
model = MhaMlpClassifier(hidden_layers=(50,), act_names="Tanh",
                         dropout_rates=None, act_output=None,
                         optim="BaseGA", optim_params={"name": "WOA", "epoch": 100, "pop_size": 20},
                         obj_name="F1S", seed=42, verbose=True,
                         lb=-2., ub=2., mode='single', n_workers=None, termination=None)
## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS"]))
