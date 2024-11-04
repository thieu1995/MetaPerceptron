Function in model object
========================

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

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
