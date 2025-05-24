
<p align="center">
<img style="width:100%;" src="https://thieu1995.github.io/post/2023-08/metaperceptron1.png" alt="MetaPerceptron"/>
</p>


---

[![GitHub release](https://img.shields.io/badge/release-2.1.0-yellow.svg)](https://github.com/thieu1995/MetaPerceptron/releases)
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


`MetaPerceptron` (Metaheuristic-optimized Multi-Layer Perceptron) is a powerful and extensible Python library that 
brings the best of both worlds: metaheuristic optimization and deep learning via Multi-Layer Perceptron (MLP). 
Whether you're working with classic Gradient Descent techniques or state-of-the-art metaheuristic algorithms 
like GA, PSO, WOA, DE, etc., `MetaPerceptron` has you covered. With `MetaPerceptron`, you can perform searches, 
feature selection, and hyperparameter tuning using the features provided by the Scikit-Learn library.

## 🚀 Features at a Glance

- 🔧 **Estimators**: `MlpRegressor`, `MlpClassifier`, `MhaMlpRegressor`, `MhaMlpClassifier`
- 📊 **Utilities**: `MhaMlpTuner`, `MhaMlpComparator`
- 🧠 **Model Zoo**:
  - 200+ Metaheuristic-trained MLP Regressors
  - 200+ Metaheuristic-trained MLP Classifiers
  - 12 Gradient Descent-trained MLP Regressors
  - 12 Gradient Descent-trained MLP Classifiers
- 📏 **67+ Performance Metrics** (47 for regression, 20 for classification)
- ⚙️ **Support**: GPU support (for GD-based models), Scikit-learn compatible API
- 📚 **Documentation**: https://metaperceptron.readthedocs.io
- 🐍 **Python**: 3.8+
- 📦 **Dependencies**: numpy, scipy, scikit-learn, pytorch, mealpy, pandas, permetrics


## 📖 Citation

If MetaPerceptron supports your work, please consider citing the following:

```bibtex
@article{van2025metaperceptron,
  title={MetaPerceptron: A Standardized Framework for Metaheuristic-Driven Multi-Layer Perceptron Optimization},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali and Garg, Harish and Hoang, Nguyen Thanh},
  journal={Computer Standards \& Interfaces},
  pages={103977},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.csi.2025.103977},
  url={https://doi.org/10.1016/j.csi.2025.103977}
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
  publisher={Elsevier},
  doi={10.1016/j.jhydrol.2022.129034}
}
```


## 🧪 Quick Start

Install via [current PyPI release](https://pypi.python.org/pypi/metaperceptron):

```bash
pip install metaperceptron
```

Check version:

```python
import metaperceptron
print(metaperceptron.__version__)
```


### ✅ Import core components

Here is how you can import all provided classes from `MetaPerceptron`

```python
from metaperceptron import DataTransformer, Data
from metaperceptron import MhaMlpRegressor, MhaMlpClassifier, MlpRegressor, MlpClassifier
from metaperceptron import MhaMlpTuner, MhaMlpComparator
```

### 🔍 Example: Training an MLP Classifier with Genetic Algorithm

In this tutorial, we will use Genetic Algorithm to train Multi-Layer Perceptron network for classification task.
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
model = MhaMlpClassifier(hidden_layers=(50, 15), act_names="Tanh", 
                         dropout_rates=None, act_output=None, 
                         optim="BaseGA", optim_params={"epoch": 100, "pop_size": 20, "name": "GA"}, 
                         obj_name="F1S", seed=42, verbose=True)
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

## 💬 Support

- 📦 [Source Code](https://github.com/thieu1995/MetaPerceptron)
- 📖 [Documentation](https://metaperceptron.readthedocs.io/)
- ⬇️ [PyPI Releases](https://pypi.org/project/metaperceptron/)
- ❗ [Report Issues](https://github.com/thieu1995/MetaPerceptron/issues)
- 📝 [Changelog](https://github.com/thieu1995/MetaPerceptron/blob/master/ChangeLog.md)
- 💬 [Chat Group](https://t.me/+fRVCJGuGJg1mNDg1)

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=MetaPerceptron_QUESTIONS) @ 2025
