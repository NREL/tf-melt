Welcome to TF-MELT's documentation!
===================================

``TF-MELT`` (TensorFlow Machine Learning Toolbox) is a collection of architectures,
processing, and utilities that are transferable over a range of ML applications.

``TF-MELT`` is a toolbox for researchers to use for machine learning applications in the
TensorFlow language. The goal of this software is to enable fast start-up of machine
learning tasks and to provide a reliable and flexible framework for development and
deployment. The toolbox contains generalized methods for every aspect of the machine
learning workflow while simultaneously providing routines that can be tailored to
specific application spaces.

The toolbox is structured with the following modules further described in the
``TF-MELT`` :ref:`Package <tfmelt-package>` section:

- ``TF-MELT`` :ref:`Models Module <tfmelt.models>` - Contains a collection of pre-built
  models that can be used for a variety of machine learning tasks.

  The models currently available are:

   - `Artificial Neural Network (ANN)` - A simple feedforward neural network with
     customizable layers and activation functions.
   - `Residual Neural Network (ResNet)` - A neural network architecture with
     customizable residual blocks.
   - `Bayesian Neural Network (BNN)` - A neural network architecture with Bayesian
     layers for probabilistic modeling.

- ``TF-MELT`` :ref:`Blocks Module <tfmelt.blocks>` - Contains a collection of pre-built
  blocks that can be used to build custom models. These blocks are designed to be easily
  imported and used in custom models. Refer to the :ref:`Models <tfmelt.models>` module
  for examples of how to use these effectively.

   The blocks currently available are:

   - `DenseBlock` - A dense block for fully-connected models.
   - `ResidualBlock` - A residual block with skip connections.
   - `BayesianBlock` - A Bayesian block using flipout layers for probabilistic modeling.
   - `DefualtOutput` - A single dense layer for output.
   - `SingleMixtureOutput` - A single dense layer with mean and variance output.
   - `MultipleMixturesOutput` - A dense layer with mixture model output for multiple
     means and variances with learnable mixture coefficients.
   - `BayesianAleatoricOutput` - A Bayesian output layer with aleatoric uncertainty.

- ``TF-MELT`` :ref:`Losses Module <tfmelt.losses>` - Contains a collection of pre-built
  loss functions that can be used for a variety of machine learning tasks. These loss
  functions are designed to be easily imported and used in custom models. Refer to the
  :ref:`Models <tfmelt.models>` module for examples of how to use these effectively.

   The loss functions currently available are:

   - `SingleMixtureLoss` - A negative log likelihood loss function for single mixture
     models.
   - `MultipleMixturesLoss` - A negative log likelihood loss function for multiple
     mixture models.

The toolbox also includes a :ref:`Utilities Subpackage <tfmelt.utils>`, which contains a
collection of functions useful for data preprocessing, model evaluation, visualization,
and other tasks.

The utility modules currently available are:

- :ref:`Evaluation Module <tfmelt.utils.evaluation>` - Contains a collection of
  functions for evaluating machine learning models. Useful for evaluating ``TF-MELT``
  model performance and extracting uncertainty quantification metrics.

- :ref:`Preprocessing Module <tfmelt.utils.preprocessing>` - Contains a collection of
  functions for preprocessing data for machine learning tasks. Leverages
  ``Scikit-learn`` preprocessing functions and implements additional helper functions.

- :ref:`Statistics Module <tfmelt.utils.statistics>` - Contains a collection of
  functions for calculating statistics and metrics for machine learning tasks. Designed
  to be utilized by the other utility functions.

- :ref:`Visualization Module <tfmelt.utils.visualization>` - Contains a collection of
  functions for visualizing data and model performance. Designed to easily generate
  plots of model performance, but can also be customized for user preferences.


Also included in the ``TF-MELT`` repo is an :ref:`Examples <examples>` directory, which
contains a set of jupyter notebooks that demonstrate how to use the different modules in
the toolbox for the full machine learning workflow (e.g., data preprocessing, model
creation, training, evaluation, and visualization).

There is also a :ref:`Tests Module <tests-module>` that contains a set of unit tests for
the different modules in the toolbox.

Finally, these docs are contained in the **Docs** directory, which can be built using
Sphinx.

Contact
=======

If you have any questions, issues, or feedback regarding ``TF-MELT``, please feel free
to contact the authors:

- Email: [nwimer@nrel.gov]
- GitHub: [https://github.com/NREL/tf-melt]

We look forward to hearing from you!


.. toctree::
   :maxdepth: 2
   :caption: Contents

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

