============
Installation
============

Library name: `MetaPerceptron`, but distributed as: `metaperceptron`. Therefore, you can


* Install the `current PyPI release <https://pypi.python.org/pypi/metaperceptron />`_::

   $ pip install metaperceptron==2.2.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/MetaPerceptron.git
   $ cd MetaPerceptron
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/MetaPerceptron


After installation, you can import MetaPerceptron as any other Python module::

   $ python
   >>> import metaperceptron
   >>> metaperceptron.__version__

========
Tutorial
========

.. include:: tutorial/provided_classes.rst
.. include:: tutorial/data_transformer.rst
.. include:: tutorial/data.rst
.. include:: tutorial/all_model_classes.rst
.. include:: tutorial/model_functions.rst
.. include:: tutorial/mha_mlp_tuner.rst
.. include:: tutorial/mha_mlp_comparator.rst


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
