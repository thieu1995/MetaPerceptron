
<p align="center">
<img style="width:100%;" src="https://thieu1995.github.io/post/2023-08/metaperceptron1.png" alt="MetaPerceptron"/>
</p>


---

[![GitHub release](https://img.shields.io/badge/release-2.0.0-yellow.svg)](https://github.com/thieu1995/MetaPerceptron/releases)
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
[![DOI](https://zenodo.org/badge/676088001.svg)](https://zenodo.org/doi/10.5281/zenodo.10251021)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MetaPerceptron (Metaheuristic-optimized Multi-Layer Perceptron) is a Python library that implements variants and the 
traditional version of Multi-Layer Perceptron models. These include Metaheuristic-optimized MLP models (GA, PSO, WOA, TLO, DE, ...) 
and Gradient Descent-optimized MLP models (SGD, Adam, Adelta, Adagrad, ...). It provides a comprehensive list of 
optimizers for training MLP models and is also compatible with the Scikit-Learn library. With MetaPerceptron, 
you can perform searches and hyperparameter tuning using the features provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: `MlpRegressor`, `MlpClassifier`, `MhaMlpRegressor`, `MhaMlpClassifier`
* **Provided Utility**: `MhaMlpTuner` and `MhaMlpComparator` 
* **Total Metaheuristic-trained MLP Regressor**: > 200 Models 
* **Total Metaheuristic-trained MLP Classifier**: > 200 Models
* **Total Gradient Descent-trained MLP Regressor**: 12 Models
* **Total Gradient Descent-trained MLP Classifier**: 12 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://metaperceptron.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, torch, mealpy, pandas, permetrics. 


# Citation Request 

If you want to understand how Metaheuristic is applied to Multi-Layer Perceptron, you need to read the paper 
titled **"Let a biogeography-based optimizer train your Multi-Layer Perceptron"**. 
The paper can be accessed at the following [link](https://doi.org/10.1016/j.ins.2014.01.038)


Please include these citations if you plan to use this library:

```code

@software{nguyen_van_thieu_2023_10251022,
  author       = {Nguyen Van Thieu},
  title        = {MetaPerceptron: A Standardized Framework for Metaheuristic-Trained Multi-Layer Perceptron},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10251021},
  url          = {https://github.com/thieu1995/MetaPerceptron}
}

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

# Simple Tutorial

* Install the [current PyPI release](https://pypi.python.org/pypi/metaperceptron):
```sh
$ pip install metaperceptron==2.0.0
```

* Check the version:

```sh
$ python
>>> import metaperceptron
>>> metaperceptron.__version__
```

* Import all provided classes from MetaPerceptron

```python
from metaperceptron import DataTransformer, Data
from metaperceptron import MhaMlpRegressor, MhaMlpClassifier, MlpRegressor, MlpClassifier
from metaperceptron import MhaMlpTuner, MhaMlpComparator
```

* In this tutorial, we will use Genetic Algorithm to train Multi-Layer Perceptron network for classification task.
For more complex examples and use cases, please check the folder [examples](examples).

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metaperceptron import DataTransformer, MhaMlpClassifier

## Load the dataset
X, y = load_iris(return_X_y=True)

## Split train and test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

## Scale dataset with two methods: standard and minmax
dt = DataTransformer(scaling_methods=("standard", "minmax"))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)

## Define Genetic Algorithm-trained Multi-Layer Perceptron
opt_paras = {"epoch": 100, "pop_size": 20}
model = MhaMlpClassifier(hidden_layers=(50,), act_names="Tanh", dropout_rates=None, act_output=None,
                         optim="BaseGA", optim_paras=opt_paras, obj_name="F1S", seed=42, verbose=True)
## Train the model
model.fit(X=X_train_scaled, y=y_train)

## Test the model
y_pred = model.predict(X_test)
print(y_pred)

## Print the score
print(model.score(X_test_scaled, y_test))

## Calculate some metrics
print(model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["AS", "PS", "RS", "F2S", "CKS", "FBS"]))
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
