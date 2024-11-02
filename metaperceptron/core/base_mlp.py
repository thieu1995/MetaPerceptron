#!/usr/bin/env python
# Created by "Thieu" at 14:17, 26/10/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import TypeVar
import numpy as np
import torch
import torch.nn as nn
from permetrics import ClassificationMetric, RegressionMetric
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from metaperceptron.helpers.metric_util import get_all_regression_metrics, get_all_classification_metrics
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
import pickle
import pandas as pd
from pathlib import Path
from metaperceptron.helpers import validator

# Create a TypeVar for the base class
EstimatorType = TypeVar('EstimatorType', bound='BaseMlp')


class EarlyStopper:
    """
    A utility class for implementing early stopping in training processes to prevent overfitting.

    Attributes:
        - patience (int): Number of consecutive epochs to tolerate no improvement before stopping.
        - epsilon (float): Minimum loss improvement threshold to reset the patience counter.
        - counter (int): Tracks the number of epochs without sufficient improvement.
        - min_loss (float): Keeps track of the minimum observed loss.
    """

    def __init__(self, patience=1, epsilon=0.01):
        """
        Initialize the EarlyStopper with specified patience and epsilon.

        Parameters:
            - patience (int): Maximum number of epochs without improvement before stopping.
            - epsilon (float): Minimum loss reduction to reset the patience counter.
        """
        self.patience = patience
        self.epsilon = epsilon
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        """
        Checks if training should be stopped based on the current loss.

        Parameters:
            - loss (float): The current loss value for the epoch.

        Returns:
            - bool: True if training should stop, False otherwise.
        """
        if loss < self.min_loss:
            # Loss has improved; reset counter and update min_loss
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.epsilon):
            # Loss did not improve sufficiently; increment counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomMLP(nn.Module):
    """
    A customizable multi-layer perceptron (MLP) model with flexible hidden layers, activations,
    and dropout rates, suitable for various tasks such as classification and regression.

    Attributes:
        - SUPPORTED_ACTIVATIONS (list of str): A list of supported activation function names.
        - network (nn.Sequential): The constructed MLP network with layers.

    Parameters:
        - size_input (int): Number of input features.
        - size_output (int): Number of output nodes.
        - hidden_layers (list of int): Number of nodes in each hidden layer.
        - act_names (list of str or nn.Module): Activation functions for each hidden layer.
        - dropout_rates (list of float): Dropout rates for each hidden layer.
        - task (str): Task type, "classification", "binary_classification", or "regression".
        - act_output (str or None): Activation function for the output layer; uses default if None.
    """

    SUPPORTED_ACTIVATIONS = [
        "Threshold", "ReLU", "RReLU", "Hardtanh", "ReLU6",
        "Sigmoid", "Hardsigmoid", "Tanh", "SiLU", "Mish", "Hardswish", "ELU",
        "CELU", "SELU", "GLU", "GELU", "Hardshrink", "LeakyReLU",
        "LogSigmoid", "Softplus", "Softshrink", "MultiheadAttention", "PReLU",
        "Softsign", "Tanhshrink", "Softmin", "Softmax", "Softmax2d", "LogSoftmax",
    ]

    def __init__(self, size_input, size_output, hidden_layers, act_names, dropout_rates,
                 task="classification", act_output=None):
        """
        Initialize a customizable multi-layer perceptron (MLP) model.
        """
        super(CustomMLP, self).__init__()

        # Ensure hidden_layers is a valid list, tuple, or numpy array
        if not isinstance(hidden_layers, (list, tuple, np.ndarray)):
            raise TypeError('hidden_layers must be a list or tuple or a numpy array.')

        # Ensure act_names is a valid list, tuple, numpy array, or str
        if not isinstance(act_names, (list, tuple, np.ndarray, str)):
            raise TypeError('act_names must be a list or tuple or a numpy array or name of activation functions.')
        else:
            if type(act_names) is str:
                act_names = [act_names] * len(hidden_layers)
            elif len(act_names) != len(hidden_layers):
                raise ValueError('if act_names is list, then len(act_names) must equal len(hidden_layers).')

        # Configure dropout rates
        if dropout_rates is None:
            dropout_rates = [0] * len(hidden_layers)
        elif isinstance(dropout_rates, (list, tuple, np.ndarray, float)):
            if type(dropout_rates) is float and 0 < dropout_rates < 1:
                dropout_rates = [dropout_rates] * len(hidden_layers)
            elif len(dropout_rates) != len(hidden_layers):
                raise ValueError('if dropout_rates is list, then len(dropout_rates) must equal len(hidden_layers).')
        else:
            raise TypeError('dropout_rates must be a list or tuple or a numpy array or float.')

        # Determine activation for the output layer based on the task
        if act_output is None:
            if task == 'classification':
                act_out = nn.Softmax(dim=1)
            elif task == 'binary_classification':
                act_out = nn.Sigmoid()
            else:  # regression
                act_out = nn.Identity()
        else:
            act_out = self._get_act(act_output)()

        # Initialize the layers
        layers = []
        in_features = size_input

        # Create each hidden layer with specified activation and dropout
        for out_features, act, dropout in zip(hidden_layers, act_names, dropout_rates):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self._get_act(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features  # Update input for next layer

        # Add output layer
        layers.append(nn.Linear(in_features, size_output))
        layers.append(act_out)

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def _get_act(self, act_name):
        """
        Retrieve the activation function by name.

        Parameters:
            - act_name (str): Name of the activation function.

        Returns:
            - nn.Module: The activation function module.
        """
        if act_name == "Softmax":
            return nn.Softmax(dim=0)
        elif act_name == "none":
            return nn.Identity()
        else:
            return getattr(nn.modules.activation, act_name)()

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Parameters:
            - x (torch.Tensor): The input tensor.

        Returns:
            - torch.Tensor: The output of the MLP model.
        """
        return self.network(x)

    def set_weights(self, solution):
        """
        Set network weights based on a given solution vector.

        Parameters:
            - solution (np.ndarray): A flat array of weights to set in the model.
        """
        with torch.no_grad():
            idx = 0
            for param in self.network.parameters():
                param_size = param.numel()
                param.copy_(torch.tensor(solution[idx:idx + param_size]).view(param.shape))
                idx += param_size

    def get_weights(self):
        """
        Retrieve network weights as a flattened array.

        Returns:
            - np.ndarray: Flattened array of the model's weights.
        """
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.network.parameters()])

    def get_weights_size(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            - int: Total number of parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


# Custom Sklearn-compatible MLP using PyTorch
class BaseMlp(BaseEstimator):
    """
    A custom MLP model compatible with sklearn, implemented using PyTorch. This class supports
    Multi-Layer Perceptron for both classification and regression tasks, with customizable
    hidden layers, activation functions, and dropout rates.

    Parameters
    ----------
    hidden_layers : list of int
        Specifies the number of nodes in each hidden layer.
    act_names : list of str
        List of activation function names, one for each hidden layer.
    dropout_rates : list of float
        Dropout rates for each hidden layer (0 indicates no dropout).
    task : str, optional
        Task type, either "classification" or "regression". Default is "classification".
    act_output : str or None, optional
        Activation function for the output layer, default depends on the task type.
    """

    def __init__(self, hidden_layers, act_names, dropout_rates, task="classification", act_output=None):
        self.hidden_layers = hidden_layers
        self.act_names = act_names
        self.dropout_rates = dropout_rates
        self.task = task
        self.act_output = act_output
        self.model = None
        self.loss_train = None

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        """
        Validates if the given method is supported.

        Parameters
        ----------
        method : str
            The method to be checked.
        list_supported_methods : list of str
            A list of supported method names.

        Returns
        -------
        bool
            True if the method is supported; otherwise, raises ValueError.
        """
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belong to {list_supported_methods}")

    def fit(self, X, y):
        """
        Train the MLP model on the given dataset.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Training features.
        y : array-like or torch.Tensor
            Target values.
        """
        pass

    def predict(self, X):
        """
        Generate predictions for input data using the trained model.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input features for prediction.

        Returns
        -------
        array-like or torch.Tensor
            Model predictions for each input sample.
        """
        pass

    def score(self, X, y):
        """
        Evaluate the model on the given dataset.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Evaluation features.
        y : array-like or torch.Tensor
            True values.

        Returns
        -------
        float
            The accuracy or evaluation score.
        """
        pass

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Evaluate regression performance metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values.
        list_metrics : tuple of str, list of str
            List of metrics for evaluation (e.g., "MSE" and "MAE").

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluate classification performance metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted labels.
        list_metrics : tuple of str, list of str
            List of metrics for evaluation (e.g., "AS" and "RS").

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """
        Evaluate the model using specified metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Model's predicted values.
        list_metrics : list of str, optional
            Names of metrics for evaluation (e.g., "MSE", "MAE").

        Returns
        -------
        dict
            Evaluation metrics and their values.
        """
        pass

    def save_training_loss(self, save_path="history", filename="loss.csv"):
        """
        Save training loss history to a CSV file.

        Parameters
        ----------
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving loss history (default: "loss.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} model doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_evaluation_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history",
                                filename="metrics.csv"):
        """
        Save evaluation metrics to a CSV file.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Model predictions.
        list_metrics : list of str, optional
            Metrics for evaluation (default: ("RMSE", "MAE")).
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving metrics (default: "metrics.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save true and predicted values to a CSV file.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input features.
        y_true : array-like
            True values.
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving predicted values (default: "y_predicted.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save the trained model to a pickle file.

        Parameters
        ----------
        save_path : str, optional
            Path to save the model (default: "history").
        filename : str, optional
            Filename for saving model, with ".pkl" extension (default: "model.pkl").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl") -> EstimatorType:
        """
        Load a model from a pickle file.

        Parameters
        ----------
        load_path : str, optional
            Path to load the model from (default: "history").
        filename : str, optional
            Filename of the saved model (default: "model.pkl").

        Returns
        -------
        BaseMlp
            The loaded model.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))

