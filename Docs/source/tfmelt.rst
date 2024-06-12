.. _tfmelt-package:

TF-MELT package
===============

TF-MELT is composed of a series of modules that form the basis for building a variety
of machine learning models. The structure of the packages is to encourage modularity
and reusability of the code. The main modules are:

- :ref:`Blocks Module <tfmelt.blocks>`: Each of the blocks is a self-contained
  TensorFlow model in itself, but can be combined with other blocks to form more complex
  models. The blocks are designed to be easily combined with other blocks, and to be
  easily extended.

- :ref:`Losses Module <tfmelt.losses>`: The losses are custom loss functions that should
  be used with their respective models. Certain loss functions are designed to be used
  with specific models, but others can be used with any model.

- :ref:`Models Module <tfmelt.models>`: The models are the main machine learning models
  that are built using the blocks and losses. The models serve a dual purpose of being a
  standalone model, and also as a template for building more complex models with the
  TF-MELT blocks.

Following is a detailed description of the modules and subpackages in the TF-MELT.


.. _tfmelt.blocks:

Blocks Module
-------------

.. automodule:: tfmelt.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. tfmelt.gp\_models module
.. ------------------------

.. .. automodule:: tfmelt.gp_models
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. _tfmelt.losses:

Losses Module
-------------

.. automodule:: tfmelt.losses
   :members:
   :undoc-members:
   :show-inheritance:

.. _tfmelt.models:

Models Module
-------------

.. automodule:: tfmelt.models
   :members:
   :undoc-members:
   :show-inheritance:


.. _tfmelt.subpackages:

Subpackages
-----------

In addition to the main modules, there are subpackages that contain various utility
functions that are used in the main modules. The subpackages are:

- :ref:`TF-MELT Utilities <tfmelt.utils>` : Contains utility functions that are used in the main modules. These
   functions contain routines for data processing, model evaluation, visualization, and
   other general-purpose functions.



.. toctree::
   :maxdepth: 1

   tfmelt.utils

.. Module contents
.. ---------------

.. .. automodule:: tfmelt
..    :members:
..    :undoc-members:
..    :show-inheritance:
