#!/usr/bin/env python
# Created by "Thieu" at 16:38, 25/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metaperceptron import Data, MlpClassifier
from sklearn.datasets import load_iris


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

print(type(data.X_train), type(data.y_train))

## Create model
model = MlpClassifier(hidden_size=25, act1_name="tanh", act2_name="sigmoid", obj_name="CEL",
                      max_epochs=10, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=True)

## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test, return_prob=True)
print(y_pred)
print(model.predict(data.X_test, return_prob=False))

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test, method="AS"))
print(model.scores(X=data.X_test, y=data.y_test, list_methods=["PS", "RS", "NPV", "F1S", "F2S"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS"]))
