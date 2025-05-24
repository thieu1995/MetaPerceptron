MhaMlpComparator
================

In this example, we will use Iris classification dataset. We compare 3 models includes `GA-MLP`, `PSO-MLP`, and `WOA-MLP`.

.. code-block:: python

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
