#!/usr/bin/env python
# Created by "Thieu" at 11:27, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metaperceptron import MlpClassifier

np.random.seed(41)


def test_MlpClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    model = MlpClassifier(hidden_layers=(30,), act_names="ReLU", dropout_rates=None, act_output=None,
                          epochs=10, batch_size=16, optim="Adam", optim_paras=None,
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True)
    model.fit(X, y)
    pred = model.predict(X)
    assert MlpClassifier.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert pred[0] in (0, 1)
