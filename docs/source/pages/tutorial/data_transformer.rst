DataTransformer class
=====================

We provide many scaler classes that you can select and make a combination of transforming your data via DataTransformer class. For example:

* You want to scale data by `Loge` and then `Sqrt` and then `MinMax`.

.. code-block:: python

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


* I want to scale data by `YeoJohnson` and then `Standard`.

.. code-block:: python

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

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
