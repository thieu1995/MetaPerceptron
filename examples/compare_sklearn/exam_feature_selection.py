#!/usr/bin/env python
# Created by "Thieu" at 11:52, 20/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from metaperceptron import MhaMlpClassifier

X, y = load_iris(return_X_y=True)

## Create model
knn = KNeighborsClassifier(n_neighbors=3)
## Run KNN-feature selector
sfs1 = SequentialFeatureSelector(knn, n_features_to_select=3)
sfs1.fit(X, y)
print(sfs1.get_support())


## Create model
woa_mlp = MhaMlpClassifier(hidden_layers=(10,), act_names="ELU", dropout_rates=None, act_output=None,
                           optim="OriginalWOA", optim_params={"epoch": 50, "pop_size": 20},
                           obj_name="F1S", seed=42, verbose=False,
                           lb=None, ub=None, mode='single', n_workers=None, termination=None)
## Run WOA-MLP-feature selector
sfs2 = SequentialFeatureSelector(woa_mlp, n_features_to_select=3)
sfs2.fit(X, y)
print(sfs2.get_support())
