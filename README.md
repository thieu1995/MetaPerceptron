
<p align="center">
<img style="width:100%;" src="https://thieu1995.github.io/post/2023-08/metaperceptron1.png" alt="MetaPerceptron"/>
</p>


---

[![GitHub release](https://img.shields.io/badge/release-1.0.1-yellow.svg)](https://github.com/thieu1995/MetaPerceptron/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/metaperceptron) 
[![PyPI version](https://badge.fury.io/py/metaperceptron.svg)](https://badge.fury.io/py/metaperceptron)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metaperceptron.svg)
![PyPI - Status](https://img.shields.io/pypi/status/metaperceptron.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/metaperceptron.svg)
[![Downloads](https://pepy.tech/badge/metaperceptron)](https://pepy.tech/project/metaperceptron)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/metaperceptron/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/metaperceptron/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/metaperceptron.svg)
[![Documentation Status](https://readthedocs.org/projects/metaperceptron/badge/?version=latest)](https://metaperceptron.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/metaperceptron.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10067995.svg)](https://doi.org/10.5281/zenodo.10067995)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MetaPerceptron (Metaheuristic-optimized Multi-Layer Perceptron) is a Python library that implement the traditional MLP models 
that trained by Gradient Descent-based optimizers (SGD, Adam, Adelta, Adagrad,...), and Metaheuristic-optimized MLP models.
It provides a comprehensive list of optimizers for training MLP models and is also compatible with the Scikit-Learn library. 
With MetaPerceptron, you can perform searches and hyperparameter tuning using the functionalities provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier
* **Total Metaheuristic-based Mlp Regression**: > 200 Models 
* **Total Metaheuristic-based Mlp Classification**: > 200 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://metaperceptron.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch, skorch


# Citation Request 

If you want to understand how Metaheuristic is applied to Multi-Layer Perceptron, you need to read the paper 
titled **"Let a biogeography-based optimizer train your Multi-Layer Perceptron"**. 
The paper can be accessed at the following [link](https://doi.org/10.1016/j.ins.2014.01.038)


Please include these citations if you plan to use this library:

```code

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@article{van2023groundwater,
  title={Groundwater level modeling using Augmented Artificial Ecosystem Optimization},
  author={Van Thieu, Nguyen and Barma, Surajit Deb and Van Lam, To and Kisi, Ozgur and Mahesha, Amai},
  journal={Journal of Hydrology},
  volume={617},
  pages={129034},
  year={2023},
  publisher={Elsevier}
}

@article{thieu2019efficient,
  title={Efficient time-series forecasting using neural network and opposition-based coral reefs optimization},
  author={Thieu Nguyen, Tu Nguyen and Nguyen, Binh Minh and Nguyen, Giang},
  journal={International Journal of Computational Intelligence Systems},
  volume={12},
  number={2},
  pages={1144--1161},
  year={2019}
}

```

# Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/metaperceptron):
```sh 
$ pip install metaperceptron==1.0.1
```

* Install directly from source code
```sh 
$ git clone https://github.com/thieu1995/MetaPerceptron.git
$ cd MetaPerceptron
$ python setup.py install
```

* In case, you want to install the development version from Github:
```sh 
$ pip install git+https://github.com/thieu1995/MetaPerceptron 
```

After installation, you can import MetaPerceptron as any other Python module:

```sh
$ python
>>> import metaperceptron
>>> metaperceptron.__version__
```

### Examples

In this section, we will explore the usage of the MetaPerceptron model with the assistance of a dataset. While all the 
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions 
to provide users with convenience and faster usage.  

#### Combine MetaPerceptron library like a normal library with scikit-learn.

```python
### Step 1: Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from metaperceptron import MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier

#### Step 2: Reading the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#### Step 3: Next, split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)

#### Step 4: Feature Scaling
scaler_X = MinMaxScaler()
scaler_X.fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

le_y = LabelEncoder()  # This is for classification problem only
le_y.fit(y)
y_train = le_y.transform(y_train)
y_test = le_y.transform(y_test)

#### Step 5: Fitting MLP-based core to the dataset

##### 5.1: Use standard MLP core for regression problem
regressor = MlpRegressor(expand_name="chebyshev", n_funcs=4, act_name="elu",
                          obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
regressor.fit(X_train, y_train)

##### 5.2: Use standard MLP core for classification problem 
classifer = MlpClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                           obj_name="BCEL", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
classifer.fit(X_train, y_train)

##### 5.3: Use Metaheuristic-based MLP core for regression problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_REG_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
model = MhaMlpRegressor(expand_name="chebyshev", n_funcs=3, act_name="elu",
                         obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
regressor.fit(X_train, y_train)

##### 5.4: Use Metaheuristic-based MLP core for classification problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
classifier = MhaMlpClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                               obj_name="NPV", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
classifier.fit(X_train, y_train)

#### Step 6: Predicting a new result
y_pred = regressor.predict(X_test)

y_pred_cls = classifier.predict(X_test)
y_pred_label = le_y.inverse_transform(y_pred_cls)

#### Step 7: Calculate metrics using score or scores functions.
print("Try my AS metric with score function")
print(regressor.score(X_test, y_test, method="AS"))

print("Try my multiple metrics with scores function")
print(classifier.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))
```

#### Utilities everything that Reflame provided

```python
### Step 1: Importing the libraries
from metaperceptron import Data, MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier
from sklearn.datasets import load_digits

#### Step 2: Reading the dataset
X, y = load_digits(return_X_y=True)
data = Data(X, y)

#### Step 3: Next, split dataset into train and test set
data.split_train_test(test_size=0.2, shuffle=True, random_state=100)

#### Step 4: Feature Scaling
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
data.y_test = scaler_y.transform(data.y_test)

#### Step 5: Fitting MLP-based core to the dataset

##### 5.1: Use standard MLP core for regression problem
regressor = MlpRegressor(expand_name="chebyshev", n_funcs=4, act_name="tanh",
                          obj_name="MSE", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
regressor.fit(data.X_train, data.y_train)

##### 5.2: Use standard MLP core for classification problem 
classifer = MlpClassifier(expand_name="chebyshev", n_funcs=4, act_name="tanh",
                           obj_name="BCEL", max_epochs=100, batch_size=32, optimizer="SGD", verbose=True)
classifer.fit(data.X_train, data.y_train)

##### 5.3: Use Metaheuristic-based MLP core for regression problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_REG_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
model = MhaMlpRegressor(expand_name="chebyshev", n_funcs=3, act_name="elu",
                         obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
regressor.fit(data.X_train, data.y_train)

##### 5.4: Use Metaheuristic-based MLP core for classification problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
classifier = MhaMlpClassifier(expand_name="chebyshev", n_funcs=4, act_name="sigmoid",
                               obj_name="NPV", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True)
classifier.fit(data.X_train, data.y_train)

#### Step 6: Predicting a new result
y_pred = regressor.predict(data.X_test)

y_pred_cls = classifier.predict(data.X_test)
y_pred_label = scaler_y.inverse_transform(y_pred_cls)

#### Step 7: Calculate metrics using score or scores functions.
print("Try my AS metric with score function")
print(regressor.score(data.X_test, data.y_test, method="AS"))

print("Try my multiple metrics with scores function")
print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))
```

A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing 
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize 
the data within a particular range.



1) Where do I find the supported metrics like above ["AS", "PS", "RS"]. What is that?
You can find it here: https://github.com/thieu1995/permetrics or use this

```python
from metaperceptron import MhaMlpClassifier, MhaMlpRegressor

print(MhaMlpRegressor.SUPPORTED_REG_OBJECTIVES)
print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)
```

2) I got this type of error

```code
raise ValueError("Existed at least one new label in y_pred.")
ValueError: Existed at least one new label in y_pred.
``` 
How to solve this?

+ This occurs only when you are working on a classification problem with a small dataset that has many classes. For 
  instance, the "Zoo" dataset contains only 101 samples, but it has 7 classes. If you split the dataset into a 
  training and testing set with a ratio of around 80% - 20%, there is a chance that one or more classes may appear 
  in the testing set but not in the training set. As a result, when you calculate the performance metrics, you may 
  encounter this error. You cannot predict or assign new data to a new label because you have no knowledge about the 
  new label. There are several solutions to this problem.

+ 1st: Use the SMOTE method to address imbalanced data and ensure that all classes have the same number of samples.

```python
import pandas as pd
from imblearn.over_sampling import SMOTE
from metaperceptron import Data

dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]

X_new, y_new = SMOTE().fit_resample(X, y)
data = Data(X_new, y_new)
```

+ 2nd: Use different random_state numbers in split_train_test() function.

```python
import pandas as pd
from metaperceptron import Data

dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)
data.split_train_test(test_size=0.2, random_state=10)  # Try different random_state value 
```


# Support (questions, problems)

### Official Links 

* Official source code repo: https://github.com/thieu1995/MetaPerceptron
* Official document: https://metapeceptron.readthedocs.io/
* Download releases: https://pypi.org/project/metaperceptron/
* Issue tracker: https://github.com/thieu1995/MetaPerceptron/issues
* Notable changes log: https://github.com/thieu1995/MetaPerceptron/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are "optimization" and "machine learning", check it here:
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/IntelELM
    * https://github.com/thieu1995/reflame
    * https://github.com/aiir-team
