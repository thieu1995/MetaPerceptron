#!/usr/bin/env python
# Created by "Thieu" at 15:10, 17/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metaperceptron import MhaMlpClassifier, Data
from sklearn.datasets import make_classification


# Create a multi-class classification dataset with 4 classes
X, y = make_classification(
    n_samples=300,  # Total number of data points
    n_features=7,  # Number of features
    n_informative=3,  # Number of informative features
    n_redundant=0,  # Number of redundant features
    n_classes=4,  # Number of classes
    random_state=42
)
data = Data(X, y, name="RandomData")
data.split_train_test(test_size=0.2, random_state=2)

opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 30}
model = MhaMlpClassifier(hidden_layers=(100,), act_names="ELU", dropout_rates=0.2, act_output=None,
                       optim="BaseGA", optim_paras=opt_paras, obj_name="F1S", seed=42, verbose=True)
model.fit(data.X_train, data.y_train)
y_pred = model.predict(data.X_test)
print(model.evaluate(data.y_test, y_pred, list_metrics=("AS", "PS", "F1S")))
model.save_training_loss(save_path="history", filename="loss.csv")
model.save_evaluation_metrics(data.y_test, y_pred, list_metrics=("AS", "PS", "F1S"), save_path="history", filename="metrics.csv")

## Save Model
model.save_model(save_path="history", filename="ga-elm.pkl")

## Load Model
new_model = MhaMlpClassifier()
trained_model = new_model.load_model(load_path="history", filename="ga-elm.pkl")
y_pred = trained_model.predict(data.X_test)
print(trained_model.evaluate(data.y_test, y_pred, list_metrics=("AS", "PS", "F1S")))
