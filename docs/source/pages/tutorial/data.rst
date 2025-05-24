Data class
==========

+ You can load your dataset into `Data` class
+ You can split dataset to train and test set
+ You can scale dataset without using `DataTransformer` class
+ You can scale labels using `LabelEncoder`

For example.

.. code-block:: python

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

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
