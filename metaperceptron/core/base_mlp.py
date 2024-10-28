#!/usr/bin/env python
# Created by "Thieu" at 14:17, 26/10/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
The structure and idea of this module is followed:

CustomMLP class

BaseMlp class: Inherit BaseEstimator from Scikit-Learn
    .Contains CustomMLP object
    .Below are subclass

    BaseStandardMlp class       - Gradient-based training
        MlpRegressor class
        MlpClassifier class

    BaseMhaMlp class            - Metaheuristic-based training
        MhaMlpRegressor class
        MhaMlpClassifier class

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from metaperceptron.helpers.metric_util import get_all_regression_metrics, get_all_classification_metrics
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar


class EarlyStopper:
    def __init__(self, patience=1, epsilon=0.01):
        self.patience = patience
        self.epsilon = epsilon
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.epsilon):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomMLP(nn.Module):

    SUPPORTED_ACTIVATIONS = [
        "Threshold","ReLU","RReLU","Hardtanh","ReLU6",
        "Sigmoid", "Hardsigmoid", "Tanh","SiLU", "Mish","Hardswish","ELU",
        "CELU", "SELU","GLU","GELU","Hardshrink", "LeakyReLU",
        "LogSigmoid", "Softplus","Softshrink", "MultiheadAttention","PReLU",
        "Softsign", "Tanhshrink","Softmin","Softmax", "Softmax2d","LogSoftmax",
    ]

    def __init__(self, size_input, size_output, hidden_layers, act_names, dropout_rates,
                 task="classification", act_output=None):
        """
        Initialize a customizable multi-layer perceptron (MLP) model.

        Parameters:
            - size_input (int): The number of input features.
            - hidden_layers (list of int): A list specifying the number of nodes in each hidden layer.
            - output_size (int): The number of output nodes.
            - act_names (list of nn.Module): A list of activation functions, one for each hidden layer.
            - dropout_rates (list of float): A list of dropout rates, one for each hidden layer (0 means no dropout).
        """
        super(CustomMLP, self).__init__()

        if not isinstance(hidden_layers, (list, tuple, np.ndarray)):
            raise TypeError('hidden_layers must be a list or tuple or a numpy array.')

        if not isinstance(act_names, (list, tuple, np.ndarray, str)):
            raise TypeError('act_names must be a list or tuple or a numpy array or name of activation functions.')
        else:
            if type(act_names) is str:
                act_names = [act_names] * len(hidden_layers)
            else:
                if len(act_names) != len(hidden_layers):
                    raise ValueError('if act_names is list, then len(act_names) must equal len(hidden_layers).')

        if dropout_rates is None:
            dropout_rates = [0] * len(hidden_layers)
        elif isinstance(dropout_rates, (list, tuple, np.ndarray, float)):
            if type(dropout_rates) is float and 0 < dropout_rates < 1:
                dropout_rates = [dropout_rates] * len(hidden_layers)
            else:
                if len(dropout_rates) != len(hidden_layers):
                    raise ValueError('if dropout_rates is list, then len(dropout_rates) must equal len(hidden_layers).')
        else:
            raise TypeError('dropout_rates must be a list or tuple or a numpy array or float.')

        # Check last activation of output
        # Output layer activation based on task type
        if act_output is None:
            if task == 'classification':
                act_out = nn.Softmax(dim=0)
            elif task == 'binary_classification':
                act_out = nn.Sigmoid()
            else:  # 'regression'
                act_out = nn.Identity()
        else:
            act_out = self._get_act(act_output)()

        # Initialize the layers
        layers = []
        in_features = size_input

        # Add each hidden layer with specified nodes, activation, and dropout
        for out_features, act, dropout in zip(hidden_layers, act_names, dropout_rates):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self._get_act(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features  # Set input for the next layer as output of current layer

        # Output layer
        layers.append(nn.Linear(in_features, size_output))
        layers.append(act_out)

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def _get_act(self, act_name):
        if act_name == "Softmax":
            act_func = nn.Softmax(dim=0)
        elif act_name == "none":
            act_func = nn.Identity()
        else:
            # act_func = getattr(nn.functional, act_name)
            act_func = getattr(nn.modules.activation, act_name)()
        return act_func

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Parameters:
            - x (torch.Tensor): The input tensor.

        Returns:
            - torch.Tensor: The output of the MLP model.
        """
        return self.network(x)

    # Flatten and set network weights from a chromosome (solution)
    def set_weights(self, solution):
        with torch.no_grad():
            idx = 0
            for param in self.network.parameters():
                param_size = param.numel()
                param.copy_(torch.tensor(solution[idx:idx + param_size]).view(param.shape))
                idx += param_size

    # Convert network weights to a chromosome (solution)
    def get_weights(self):
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.network.parameters()])


# Custom Sklearn-compatible MLP using PyTorch
class BaseMlp(BaseEstimator):
    def  __init__(self, hidden_layers, act_names, dropout_rates, task="classification", act_output=None):
        self.hidden_layers = hidden_layers
        self.act_names = act_names
        self.dropout_rates = dropout_rates
        self.task = task
        self.act_output = act_output

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class BaseStandardMlp(BaseMlp):

    SUPPORTED_OPTIMIZERS = [
        "Adafactor", "Adadelta", "Adagrad", "Adam",
        "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam",
        "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    ]

    def __init__(self, hidden_layers=(100, ), act_names="ReLU", dropout_rates=0.2, act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        super().__init__(hidden_layers, act_names, dropout_rates, "classification", act_output)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_paras = optim_paras
        if optim_paras is None:
            self.optim_paras = {}
        self.early_stopping = early_stopping
        self.n_patience = n_patience
        self.epsilon = epsilon
        self.valid_rate = valid_rate
        self.seed = seed
        self.verbose = verbose

        self.size_input = None
        self.size_output = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.patience_count = None
        self.valid_mode = False
        self.early_stopper = None

    def build_model(self):
        if self.early_stopping:
            self.early_stopper = EarlyStopper(patience=self.n_patience, epsilon=self.epsilon)

        self.model = CustomMLP(self.size_input, self.size_output, self.hidden_layers, self.act_names,
                               self.dropout_rates, self.task, self.act_output)
        self.optimizer = getattr(torch.optim, self.optim)(self.model.parameters(), **self.optim_paras)
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:       # regression or multi_regression
            self.criterion = nn.MSELoss()

    def process_data(self, X, y, **kwargs):
        pass

    def _fit(self, data, **kwargs):
        # Get data
        train_loader, X_valid_tensor, y_valid_tensor = data

        # Training loop
        self.model.train()  # Set model to training mode
        for epoch in range(self.epochs):

            # Training step
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()  # Clear gradients

                # Forward pass
                output = self.model(batch_X)
                # Compute loss based on classification type
                loss = self.criterion(output, batch_y)

                # Backward pass and optimizer step
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  # Accumulate batch loss

            # Average loss per epoch
            avg_loss = total_loss / len(train_loader)

            # Validation step
            if self.valid_mode:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_valid_tensor)
                    val_loss = self.criterion(val_output, y_valid_tensor)

                if self.early_stopping:
                    if self.early_stopper.early_stop(val_loss):
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                if self.early_stopping:
                    if self.early_stopper.early_stop(avg_loss):
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}")

            # Set back to training mode
            self.model.train()


class MlpClassifier(BaseStandardMlp, ClassifierMixin):
    def __init__(self, hidden_layers=(100, ), act_names="ReLU", dropout_rates=0.2, act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        super().__init__(hidden_layers, act_names, dropout_rates, act_output,
                         epochs, batch_size, optim, optim_paras,
                         early_stopping, n_patience, epsilon, valid_rate, seed, verbose)
        self.classes_ = None

    def process_data(self, X, y, **kwargs):
        X_valid_tensor, y_valid_tensor = None, None
        # Convert data to tensors and create a DataLoader for batch processing
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Split data into training and validation sets
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True, stratify=y)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if self.task == "binary_classification":
            y_tensor = torch.tensor(y, dtype=torch.float32)
            y_tensor = torch.unsqueeze(y_tensor, 1)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
            if self.task == "binary_classification":
                y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
                y_valid_tensor = torch.unsqueeze(y_valid_tensor, 1)
        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y, **kwargs):
        ## Check the parameters
        self.size_input = X.shape[1]
        y = np.squeeze(np.array(y))
        if y.ndim != 1:
            y = np.argmax(y, axis=1)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.task = "binary_classification"
            self.size_output = 1
        else:
            self.task = "classification"
            self.size_output = len(self.classes_)

        ## Process data
        data = self.process_data(X, y, **kwargs)

        ## Build model
        self.build_model()

        ## Fit the data
        self._fit(data, **kwargs)

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class MlpRegressor(BaseStandardMlp, RegressorMixin):
    def __init__(self, hidden_layers=(100, ), act_names="ELU", dropout_rates=0.2, act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        super().__init__(hidden_layers, act_names, dropout_rates, act_output,
                         epochs, batch_size, optim, optim_paras,
                         early_stopping, n_patience, epsilon, valid_rate, seed, verbose)

    def process_data(self, X, y, **kwargs):
        X_valid_tensor, y_valid_tensor = None, None
        # Convert data to tensors and create a DataLoader for batch processing
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Split data into training and validation sets
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
            if y_valid_tensor.ndim == 1:
                y_valid_tensor = y_valid_tensor.unsqueeze(1)
        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y, **kwargs):
        ## Check the parameters
        self.size_input = X.shape[1]
        y = np.squeeze(np.array(y))
        self.size_output = 1
        self.task = "regression"
        if y.ndim == 2:
            self.task = "multi_regression"
            self.size_output = y.shape[1]

        ## Process data
        data = self.process_data(X, y, **kwargs)

        ## Build model
        self.build_model()

        ## Fit the data
        self._fit(data, **kwargs)

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(X_tensor)
        return predicted.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class BaseMhaMlp(BaseMlp):

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()
    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    SUPPORTED_ACTIVATIONS = ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu",
                             "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "log_sigmoid", "swish",
                             "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink",
                             "hard_shrink", "softmin", "softmax", "log_softmax", "silu"]
    CLS_OBJ_LOSSES = None

    def __init__(self, hidden_layers=(100, ), act_names="ELU", dropout_rates=0.2, act_output=None,
                 optim="BaseGA", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        super().__init__(hidden_layers, act_names, dropout_rates, "classification", act_output)
        self.optim = optim
        self.optim_paras = optim_paras
        if optim_paras is None:
            self.optim_paras = {}
        self.early_stopping = early_stopping
        self.n_patience = n_patience
        self.epsilon = epsilon
        self.valid_rate = valid_rate
        self.seed = seed
        self.verbose = verbose

        self.size_input = None
        self.size_output = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.patience_count = None
        self.valid_mode = False
        self.early_stopper = None

    def _set_optimizer(self, optim=None, optim_paras=None):
        if type(optim) is str:
            opt_class = get_optimizer_by_name(optim)
            if type(optim_paras) is dict:
                return opt_class(**optim_paras)
            else:
                return opt_class(epoch=100, pop_size=50)
        elif isinstance(optim, Optimizer):
            if type(optim_paras) is dict:
                return optim.set_parameters(optim_paras)
            return optim
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def build_model(self):
        if self.early_stopping:
            self.early_stopper = EarlyStopper(patience=self.n_patience, epsilon=self.epsilon)

        self.model = CustomMLP(self.size_input, self.size_output, self.hidden_layers, self.act_names,
                               self.dropout_rates, self.task, self.act_output)



        self.optimizer = self._set_optimizer(self.optim, self.optim_paras)

        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:       # regression or multi_regression
            self.criterion = nn.MSELoss()

    def process_data(self, X, y, **kwargs):
        pass

    def _fit(self, data, **kwargs):
        # Get data
        train_loader, X_valid_tensor, y_valid_tensor = data

        # Training loop
        self.model.train()  # Set model to training mode
        for epoch in range(self.epochs):

            # Training step
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()  # Clear gradients

                # Forward pass
                output = self.model(batch_X)
                # Compute loss based on classification type
                loss = self.criterion(output, batch_y)

                # Backward pass and optimizer step
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  # Accumulate batch loss

            # Average loss per epoch
            avg_loss = total_loss / len(train_loader)

            # Validation step
            if self.valid_mode:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_valid_tensor)
                    val_loss = self.criterion(val_output, y_valid_tensor)

                if self.early_stopping:
                    if self.early_stopper.early_stop(val_loss):
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                if self.early_stopping:
                    if self.early_stopper.early_stop(avg_loss):
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}")

            # Set back to training mode
            self.model.train()


class MhaMlpClassifier(BaseMhaMlp, ClassifierMixin):
    def __init__(self, hidden_layers=(100, ), act_names="ReLU", dropout_rates=0.2, act_output=None,
                 optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        super().__init__(hidden_layers, act_names, dropout_rates, act_output, optim, optim_paras,
                         early_stopping, n_patience, epsilon, valid_rate, seed, verbose)
        self.classes_ = None

    def process_data(self, X, y, **kwargs):
        X_valid_tensor, y_valid_tensor = None, None
        # Convert data to tensors and create a DataLoader for batch processing
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Split data into training and validation sets
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True, stratify=y)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if self.task == "binary_classification":
            y_tensor = torch.tensor(y, dtype=torch.float32)
            y_tensor = torch.unsqueeze(y_tensor, 1)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
            if self.task == "binary_classification":
                y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
                y_valid_tensor = torch.unsqueeze(y_valid_tensor, 1)
        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y, **kwargs):
        ## Check the parameters
        self.size_input = X.shape[1]
        y = np.squeeze(np.array(y))
        if y.ndim != 1:
            y = np.argmax(y, axis=1)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.task = "binary_classification"
            self.size_output = 1
        else:
            self.task = "classification"
            self.size_output = len(self.classes_)

        ## Process data
        data = self.process_data(X, y, **kwargs)

        ## Build model
        self.build_model()

        ## Fit the data
        self._fit(data, **kwargs)

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

