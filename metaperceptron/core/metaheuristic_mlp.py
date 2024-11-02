#!/usr/bin/env python
# Created by "Thieu" at 17:43, 13/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import torch
from permetrics import ClassificationMetric, RegressionMetric
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import ClassifierMixin, RegressorMixin
from metaperceptron.core.base_mlp import BaseMhaMlp


class MhaMlpClassifier(BaseMhaMlp, ClassifierMixin):
    """
    A Metaheuristic-based MLP Classifier that extends the BaseMhaMlp class and implements
    the ClassifierMixin interface from Scikit-Learn for classification tasks.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique classes found in the target variable.
    metric_class : type
        The metric class used for evaluating classification performance.

    Parameters
    ----------
    hidden_layers : tuple of int, optional
        The structure of the hidden layers (default is (100,)).
    act_names : str, optional
        Activation function name to use in hidden layers (default is "ReLU").
    dropout_rates : float, optional
        Dropout rate for regularization (default is 0.2).
    act_output : any, optional
        Activation function for the output layer (default is None).
    optim : str, optional
        The optimization algorithm to use (default is "BaseGA").
    optim_paras : dict, optional
        Parameters for the optimizer (default is None).
    obj_name : str, optional
        The objective name for the optimization (default is "F1S").
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        Whether to print detailed logs during fitting (default is True).
    """

    def __init__(self, hidden_layers=(100,), act_names="ReLU", dropout_rates=0.2, act_output=None,
                 optim="BaseGA", optim_paras=None, obj_name="F1S", seed=42, verbose=True):
        """
        Initializes the MhaMlpClassifier with specified parameters.
        """
        super().__init__(hidden_layers, act_names, dropout_rates, act_output,
                         optim, optim_paras, obj_name, seed, verbose)
        self.classes_ = None  # Initialize classes to None
        self.metric_class = ClassificationMetric  # Set the metric class for evaluation

    def fit(self, X, y, lb=(-1.0,), ub=(1.0,), mode='single', n_workers=None,
            termination=None, save_population=False, **kwargs):
        """
        Fits the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        lb : tuple, optional
            Lower bounds for optimization (default is (-1.0,)).
        ub : tuple, optional
            Upper bounds for optimization (default is (1.0,)).
        mode : str, optional
            Mode for optimization (default is 'single').
        n_workers : int, optional
            Number of workers for parallel processing (default is None).
        termination : any, optional
            Termination criteria for optimization (default is None).
        save_population : bool, optional
            Whether to save the population during optimization (default is False).
        **kwargs : additional parameters
            Additional parameters for fitting.

        Returns
        -------
        self : MhaMlpClassifier
            Returns the instance of the fitted model.
        """
        ## Check the parameters
        self.size_input = X.shape[1]  # Number of features
        y = np.squeeze(np.array(y))  # Convert y to a numpy array and squeeze dimensions
        if y.ndim != 1:
            y = np.argmax(y, axis=1)  # Convert to 1D if it’s not already
        self.classes_ = np.unique(y)  # Get unique classes from y
        if len(self.classes_) == 2:
            self.task = "binary_classification"  # Set task for binary classification
            self.size_output = 1  # Output size for binary classification
        else:
            self.task = "classification"  # Set task for multi-class classification
            self.size_output = len(self.classes_)  # Output size for multi-class

        ## Process data
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor

        ## Build model
        self.build_model()  # Build the model architecture

        ## Fit the data
        self._fit((X_tensor, y), lb, ub, mode, n_workers, termination, save_population, **kwargs)  # Fit the model

        return self  # Return the fitted model

    def predict(self, X):
        """
        Predicts the class labels for the provided input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted class labels for each sample.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = self.model(X_tensor)  # Get model predictions
            _, predicted = torch.max(output, 1)  # Get the predicted class labels
        return predicted.numpy()  # Return as a numpy array

    def score(self, X, y):
        """
        Computes the accuracy score of the model based on predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for scoring.
        y : array-like, shape (n_samples,)
            True labels for comparison.

        Returns
        -------
        float
            Accuracy score of the model.
        """
        y_pred = self.predict(X)  # Get predictions
        return accuracy_score(y, y_pred)  # Calculate and return accuracy score

    def predict_proba(self, X):
        """
        Computes the probability estimates for each class (for classification tasks only).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for which to predict probabilities.

        Returns
        -------
        np.ndarray
            Probability predictions for each class.

        Raises
        ------
        ValueError
            If the task is not a classification task.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor
        if self.task not in ["classification", "binary_classification"]:
            raise ValueError(
                "predict_proba is only available for classification tasks.")  # Raise error if task is invalid

        self.model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            probs = self.model.forward(X_tensor)  # Get the output from forward pass
        return probs.numpy()  # Return probabilities as a numpy array

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            List of metrics to compute using Permetrics library:
            https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            A dictionary containing the results of the requested metrics.
        """
        return self._BaseMlp__evaluate_cls(y_true, y_pred, list_metrics)  # Call evaluation method

