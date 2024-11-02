#!/usr/bin/env python
# Created by "Thieu" at 11:25, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metaperceptron import MlpRegressor

np.random.seed(42)


def test_MlpRegressor_class():
    X = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(100, 5))
    y = 2 * X + 1 + noise

    model = MlpRegressor(hidden_layers=(30,), act_names="Tanh", dropout_rates=None, act_output=None,
                         epochs=10, batch_size=16, optim="Adam", optim_paras=None,
                         early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                         seed=42, verbose=True)
    model.fit(X, y)

    pred = model.predict(X)
    assert MlpRegressor.SUPPORTED_REG_METRICS == model.SUPPORTED_REG_METRICS
    assert len(pred) == X.shape[0]
