#!/usr/bin/env python
# Created by "Thieu" at 09:15, 24/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from metaperceptron import DataTransformer, MlpClassifier


def get_cross_val_score(X, y, cv=3):
    ## Train and test
    model = MlpClassifier(hidden_layers=(30,), act_names="ReLU", dropout_rates=None, act_output=None,
                          epochs=10, batch_size=16, optim="Adam", optim_params=None,
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True, device="cpu")
    return cross_val_score(model, X, y, cv=cv)


def get_pipe_line(X, y):
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    ## Train and test
    model = MlpClassifier(hidden_layers=(30,), act_names="ReLU", dropout_rates=None, act_output=None,
                          epochs=10, batch_size=16, optim="Adam", optim_params=None,
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True, device="cpu")

    pipe = Pipeline([
        ("dt", DataTransformer(scaling_methods=("standard", "minmax"))),
        ("pnn", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "AS", "RS", "PS"])


def get_grid_search(X, y):
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    para_grid = {
        'act_names': ("ReLU", "Tanh", "Sigmoid"),
        'hidden_layers': [(10,),  (20,), (30,) ]
    }

    ## Create a gridsearch
    model = MlpClassifier(dropout_rates=None, act_output=None,
                          epochs=10, batch_size=16, optim="Adam", optim_params=None,
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True, device="cpu")
    clf = GridSearchCV(model, para_grid, cv=3, scoring='accuracy', verbose=2)
    clf.fit(X_train, y_train)
    print("Best parameters found: ", clf.best_params_)
    print("Best model: ", clf.best_estimator_)
    print("Best training score: ", clf.best_score_)
    print(clf)

    ## Predict
    y_pred = clf.predict(X_test)
    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "AS", "RS", "PS"])


## Load data object
X, y = load_breast_cancer(return_X_y=True)

print(get_cross_val_score(X, y, cv=3))
print(get_pipe_line(X, y))
print(get_grid_search(X, y))
