============
Installation
============

Library name: `MetaPerceptron`, but distributed as: `metaperceptron`. Therefore, you can


* Install the `current PyPI release <https://pypi.python.org/pypi/metaperceptron />`_::

   $ pip install metaperceptron==2.0.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/MetaPerceptron.git
   $ cd MetaPerceptron
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/MetaPerceptron


After installation, you can import MetaPerceptron as any other Python module::

   $ python
   >>> import metaperceptron
   >>> metaperceptron.__version__


================
Provided Classes
================

* `Data`: A class for managing and structuring data. It includes methods for loading, splitting, and scaling datasets for neural network models.

* `DataTransformer`: A transformer class that applies preprocessing operations, such as scaling, normalization, or encoding, to prepare data for neural network models.

* `MlpRegressor`: A standard multi-layer perceptron (MLP) model for regression tasks. It includes configurable parameters for the number and size of hidden layers, activation functions, learning rate, and optimizer.

* `MlpClassifier`: A standard multi-layer perceptron (MLP) model for classification tasks. This model supports flexible architectures with customizable hyperparameters for improved classification performance.

* `MhaMlpRegressor`: An MLP regressor model enhanced with Meta-Heuristic Algorithms (MHA), designed to optimize training by applying metaheuristic techniques (such as Genetic Algorithms or Particle Swarm Optimization) to find better network weights and hyperparameters for complex regression tasks.

* `MhaMlpClassifier`: A classification model that combines MLP with Meta-Heuristic Algorithms (MHA) to improve training efficiency. This approach allows for robust exploration of the optimization landscape, which is beneficial for complex, high-dimensional classification problems.

* `MhaMlpTuner`: A tuner class designed to optimize MLP model hyperparameters using Meta-Heuristic Algorithms (MHA). This class leverages algorithms like Genetic Algorithms, Particle Swarm Optimization, and others to automate the tuning of parameters such as learning rate, number of hidden layers, and neuron configuration, aiming to achieve optimal model performance.

* `MhaMlpComparator`: A comparator class for evaluating and comparing the performance of different MLP models or configurations, particularly useful for assessing the impact of various Meta-Heuristic Algorithms (MHAs) on model training. The comparator allows side-by-side performance evaluation of models with distinct hyperparameter settings or MHA enhancements.


---------------------
DataTransformer class
---------------------

We provide many scaler classes that you can select and make a combination of transforming your data via DataTransformer class. For example:

* You want to scale data by `Loge` and then `Sqrt` and then `MinMax`::

    from metaperceptron import DataTransformer
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:5].values
    y = dataset.iloc[:, 5].values
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    dt = DataTransformer(scaling_methods=("loge", "sqrt", "minmax"))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)


* I want to scale data by `YeoJohnson` and then `Standard`::

    from metaperceptron import DataTransformer
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:5].values
    y = dataset.iloc[:, 5].values
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    dt = DataTransformer(scaling_methods=("yeo-johnson", "standard"))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)


----------
Data class
----------

+ You can load your dataset into `Data` class
+ You can split dataset to train and test set
+ You can scale dataset without using `DataTransformer` class
+ You can scale labels using `LabelEncoder`

For example::

    from metaperceptron import Data
    import pandas as pd

    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:5].values
    y = dataset.iloc[:, 5].values

    ## Create data object
    data = Data(X, y, name="position_salaries")

    ## Split dataset into train and test set
    data.split_train_test(test_size=0.2, shuffle=True, random_state=100, inplace=True)

    ## Feature Scaling
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "sqrt", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)

    data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
    data.y_test = scaler_y.transform(data.y_test)

------------------------
Define all model classes
------------------------

Here is how you define all of provided classes.::

    from metaperceptron import MhaMlpRegressor, MhaMlpClassifier, MlpRegressor, MlpClassifier

    ## Use Metaheuristic Algorithm-trained MLP model for regression problem
    print(MhaMlpRegressor.SUPPORTED_OPTIMIZERS)
    print(MhaMlpRegressor.SUPPORTED_REG_OBJECTIVES)

    opt_paras = {"epoch": 250, "pop_size": 30, "name": "GA"}
    model = MhaMlpRegressor(hidden_layers=(30, 15,), act_names="ELU", dropout_rates=0.2, act_output=None,
                            optim="BaseGA", optim_paras=opt_paras, obj_name="MSE", seed=42, verbose=True)


    ## Use Metaheuristic Algorithm-trained MLP model for classification problem
    print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
    print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)

    opt_paras = {"epoch": 250, "pop_size": 30, "name": "WOA"}
    model = MhaMlpClassifier(hidden_layers=(100, 20), act_names="ReLU", dropout_rates=None, act_output=None,
                             optim="OriginalWOA", optim_paras=opt_paras, obj_name="F1S", seed=42, verbose=True)


    ## Use Gradient Descent-trained (Adam Optimizer) to train MLP model for regression problem
    print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)

    model = MlpRegressor(hidden_layers=(30, 10), act_names="Tanh", dropout_rates=None, act_output=None,
                         epochs=100, batch_size=16, optim="Adagrad", optim_paras=None,
                         early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                         seed=42, verbose=True)


    ## Use Gradient Descent-trained (Adam Optimizer) to train MLP model for classification problem
    print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)

    model = MlpClassifier(hidden_layers=(30, 20), act_names="ReLU", dropout_rates=None, act_output=None,
                          epochs=100, batch_size=16, optim="Adam", optim_paras=None,
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True)

------------------------
Function in model object
------------------------

After you define model, here are several functions you can call in model object::

    from metaperceptron import MhaMlpRegressor, Data

    data = Data(X, y)       # Assumption that you have provide this object like above

    model = MhaMlpRegressor(...)

    ## Train the model
    model.fit(data.X_train, data.y_train)

    ## Predicting a new result
    y_pred = model.predict(data.X_test)

    ## Calculate metrics using score or scores functions.
    print(model.score(data.X_test, data.y_test))

    ## Calculate metrics using evaluate function
    print(model.evaluate(data.y_test, y_pred, list_metrics=("MAPE", "NNSE", "KGE", "MASE", "R2", "R", "R2S")))

    ## Save performance metrics to csv file
    model.save_evaluation_metrics(data.y_test, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv")

    ## Save training loss to csv file
    model.save_training_loss(save_path="history", filename="loss.csv")

    ## Save predicted label
    model.save_y_predicted(X=data.X_test, y_true=data.y_test, save_path="history", filename="y_predicted.csv")

    ## Save model
    model.save_model(save_path="history", filename="traditional_mlp.pkl")

    ## Load model
    trained_model = MlpRegressor.load_model(load_path="history", filename="traditional_mlp.pkl")


-----------------
MhaMlpTuner class
-----------------

In this example, we use Genetic Algorithm-trained MLP network for Breast Cancer classification dataset. We tune several hyper-paramaters of both network structure and optimizer's parameters.::

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_breast_cancer
    from metaperceptron import MhaMlpTuner

    ## Load data object
    X, y = load_breast_cancer(return_X_y=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, X_test.shape)

    # Example parameter grid of tuning hyper-parameter for Genetic Algorithm-based MLP
    param_dict = {
        'hidden_layers': [(10,), (20, 10)],
        'act_names': ['Tanh', 'ELU'],
        'dropout_rates': [0.2, None],
        'optim': ['BaseGA'],
        'optim_paras': [
            {"epoch": 10, "pop_size": 20},
            {"epoch": 20, "pop_size": 20},
        ],
        'obj_name': ["F1S"],
        'seed': [42],
        "verbose": [False],
    }

    # Initialize the tuner
    tuner = MhaMlpTuner(
        task="classification",
        param_dict=param_dict,
        search_method="randomsearch",  # or "gridsearch"
        scoring='accuracy',
        cv=3,
        verbose=2,          # Example additional argument
        random_state=42,    # Additional parameter for RandomizedSearchCV
        n_jobs=4            # Parallelization
    )

    # Perform tuning
    tuner.fit(X_train, y_train)
    print("Best Parameters: ", tuner.best_params_)
    print("Best Estimator: ", tuner.best_estimator_)

    y_pred = tuner.predict(X_test)
    print(tuner.best_estimator_.evaluate(y_test, y_pred, list_metrics=["AS", "PS", "RS", "F1S", "NPV"]))


----------------
MhaMlpComparator
----------------

In this example, we will use Iris classification dataset. We compare 3 models includes `GA-MLP`, `PSO-MLP`, and `WOA-MLP`.::

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    from metaperceptron import MhaMlpComparator

    ## Load data object
    X, y = load_iris(return_X_y=True)

    ## Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, X_test.shape)

    ## Here is the list of optimizers you want to compare
    optim_dict = {
        "BaseGA": {"epoch": 100, "pop_size": 20},
        "OriginalPSO": {"epoch": 100, "pop_size": 20},
        "OriginalWOA": {"epoch": 100, "pop_size": 20},
    }

    ## Initialize the comparator
    compartor = MhaMlpComparator(
        optim_dict=optim_dict,
        task="classification",
        hidden_layers=(30, ),
        act_names="ReLU",
        dropout_rates=None,
        act_output=None,
        obj_name="F1S",
        verbose=True,
        seed=42,
    )

    ## Perform comparison

    ## You can perform cross validation score method
    results = compartor.compare_cross_val_score(X_train, y_train, metric="AS", cv=4, n_trials=2, to_csv=True)
    print(results)

    ## Or you can perform cross validation method
    results = compartor.compare_cross_validate(X_train, y_train, metrics=["AS", "PS", "F1S", "NPV"],
                                               cv=4, return_train_score=True, n_trials=2, to_csv=True)
    print(results)

    ## Or you can perform train and test method
    results = compartor.compare_train_test(X_train, y_train, X_test, y_test,
                                           metrics=["AS", "PS", "F1S", "NPV"], n_trials=2, to_csv=True)
    print(results)


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
