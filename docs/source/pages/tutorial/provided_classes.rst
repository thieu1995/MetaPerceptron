Provided Classes
================

* `Data`: A class for managing and structuring data. It includes methods for loading, splitting, and scaling datasets for neural network models.

* `DataTransformer`: A transformer class that applies preprocessing operations, such as scaling, normalization, or encoding, to prepare data for neural network models.

* `MlpRegressor`: A standard multi-layer perceptron (MLP) model for regression tasks. It includes configurable parameters for the number and size of hidden layers, activation functions, learning rate, and optimizer.

* `MlpClassifier`: A standard multi-layer perceptron (MLP) model for classification tasks. This model supports flexible architectures with customizable hyperparameters for improved classification performance.

* `MhaMlpRegressor`: An MLP regressor model enhanced with Meta-Heuristic Algorithms (MHA), designed to optimize training by applying metaheuristic techniques (such as Genetic Algorithms or Particle Swarm Optimization) to find better network weights and hyperparameters for complex regression tasks.

* `MhaMlpClassifier`: A classification model that combines MLP with Meta-Heuristic Algorithms (MHA) to improve training efficiency. This approach allows for robust exploration of the optimization landscape, which is beneficial for complex, high-dimensional classification problems.

* `MhaMlpTuner`: A tuner class designed to optimize MLP model hyperparameters using Meta-Heuristic Algorithms (MHA). This class leverages algorithms like Genetic Algorithms, Particle Swarm Optimization, and others to automate the tuning of parameters such as learning rate, number of hidden layers, and neuron configuration, aiming to achieve optimal model performance.

* `MhaMlpComparator`: A comparator class for evaluating and comparing the performance of different MLP models or configurations, particularly useful for assessing the impact of various Meta-Heuristic Algorithms (MHAs) on model training. The comparator allows side-by-side performance evaluation of models with distinct hyperparameter settings or MHA enhancements.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
