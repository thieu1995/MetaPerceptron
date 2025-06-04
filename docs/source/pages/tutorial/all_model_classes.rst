Define all model classes
========================

Here is how you define all of provided classes.

.. code-block:: python

    from metaperceptron import MhaMlpRegressor, MhaMlpClassifier, MlpRegressor, MlpClassifier

    ## Use Metaheuristic Algorithm-trained MLP model for regression problem
    print(MhaMlpRegressor.SUPPORTED_OPTIMIZERS)
    print(MhaMlpRegressor.SUPPORTED_REG_OBJECTIVES)

    opt_paras = {"epoch": 250, "pop_size": 30, "name": "GA"}
    model = MhaMlpRegressor(hidden_layers=(30, 15,), act_names="ELU", dropout_rates=0.2, act_output=None,
                            optim="BaseGA", optim_params=opt_paras, obj_name="MSE", seed=42, verbose=True,
                            lb=None, ub=None, mode='single', n_workers=None, termination=None)


    ## Use Metaheuristic Algorithm-trained MLP model for classification problem
    print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
    print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)

    opt_paras = {"epoch": 250, "pop_size": 30, "name": "WOA"}
    model = MhaMlpClassifier(hidden_layers=(100, 20), act_names="ReLU", dropout_rates=None, act_output=None,
                             optim="OriginalWOA", optim_params=opt_paras, obj_name="F1S", seed=42, verbose=True,
                             lb=None, ub=None, mode='single', n_workers=None, termination=None)


    ## Use Gradient Descent-trained (Adam Optimizer) to train MLP model for regression problem
    print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)

    model = MlpRegressor(hidden_layers=(30, 10), act_names="Tanh", dropout_rates=None, act_output=None,
                         epochs=100, batch_size=16, optim="Adagrad", optim_params=None,
                         early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                         seed=42, verbose=True)


    ## Use Gradient Descent-trained (Adam Optimizer) to train MLP model for classification problem
    print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)

    model = MlpClassifier(hidden_layers=(30, 20), act_names="ReLU", dropout_rates=None, act_output=None,
                          epochs=100, batch_size=16, optim="Adam", optim_params=None,
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True)

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
