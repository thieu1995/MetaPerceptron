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

    model = MlpClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                          max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
    model.fit(X, y)
    pred = model.predict(X)
    assert MlpClassifier.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert pred[0] in (0, 1)
