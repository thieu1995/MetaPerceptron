#!/usr/bin/env python
# Created by "Thieu" at 09:49, 25/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metaperceptron import MhaMlpClassifier, Data
from sklearn.datasets import make_classification


# Create a multi-class classification dataset with 4 classes
X, y = make_classification(
    n_samples=300,  # Total number of data points
    n_features=7,  # Number of features
    n_informative=3,  # Number of informative features
    n_redundant=0,  # Number of redundant features
    n_classes=4,  # Number of classes
    random_state=42
)
data = Data(X, y, name="RandomData")
data.split_train_test(test_size=0.2, random_state=2)

opt_paras = {"name": "WOA", "epoch": 30, "pop_size": 30}
model = MhaMlpClassifier(hidden_layers=(100,), act_names="ELU", dropout_rates=0.2, act_output=None,
                       optim="BaseGA", optim_paras=opt_paras, obj_name="F1S", seed=42, verbose=True)
model.fit(data.X_train, data.y_train, lb=(-10., ), ub=(10., ))
y_pred = model.predict(data.X_test)

## Get parameters for model
print(model.get_params())

## Get weights of neural network
print(model.network.get_weights())

print(model.network)
