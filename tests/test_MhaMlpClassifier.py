#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metaperceptron import MhaMlpClassifier

np.random.seed(41)


def test_MhaMlpClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaMlpClassifier(hidden_layers=(100,), act_names="ELU", dropout_rates=None, act_output=None,
                             optim="BaseGA", optim_paras=opt_paras, obj_name="F1S", seed=42, verbose=True)
    model.fit(X, y)
    pred = model.predict(X)
    assert MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES == model.SUPPORTED_CLS_OBJECTIVES
    assert pred[0] in (0, 1)
