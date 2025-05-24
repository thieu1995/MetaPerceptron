
# Version 2.1.0

+ Fix bug name of optimizer in `MhaMlp` models
+ Fix bug encoding labels data in `Data` class
+ Add GPU support for Gradient Descent-trained MLP models
+ Rename `optim_paras` to `optim_params` in all models.
+ Fix bug in `validator` module
+ Fix bug missing loss_train in Gradient Descent-trained MLP models
+ Update examples, tests, documents, citations, and readme.

----------------------------------------------------------------------------------------

# Version 2.0.0

+ Re-structure and re-implement the MLP, Metaheuristic-trained MLP, and Gradient Descent-trained MLP.
+ Remove the dependence of Skorch library.
+ Add `MhaMlpTuner` class: It can tune the hyper-parameters of Metaheuristic-based MLP models.
+ Add `MhaMlpComparator` class: It can compare several Metaheuristic-based MLP models.
+ Update examples, tests, and documents
+ Update requirements, citations, readme.

----------------------------------------------------------------------------------------

# Version 1.1.0

+ Add `MhaMlpRegressor` and `MhaMlpClassifier` classes
+ Add docs folder and document website
+ Update examples and tests folders
+ Add zenodo DOI, CITATION.cff

----------------------------------------------------------------------------------------

# Version 1.0.1

+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, requirements.txt)
+ Add helpers modules (act_util, metric_util, scaler_util, validator, preprocessor)
+ Add MlpRegressor and MlpClassifier classes (Based on Pytorch and Skorch)
+ Add publish workflow
+ Add examples and tests folders
