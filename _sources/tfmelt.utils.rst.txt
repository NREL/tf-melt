.. _tfmelt.utils:

TF-MELT Utilities Subpackage
============================

This package contains utility functions for data processing, evaluation, visualization,
and other common tasks in machine learning. The subpackage is organized into the
following modules:

- :ref:`Evaluation Module <tfmelt.utils.evaluation>`: Functions for evaluating model
  performance. Includes functions for making predictions for standard and ensemble
  models.

- :ref:`Preprocessing Module <tfmelt.utils.preprocessing>`: Functions for data
  preprocessing. Includes functions for normalizing data built on `Scikit-learn`
  normalizers.

- :ref:`Statistics Module <tfmelt.utils.statistics>`: Functions for computing statistics
  on data. Includes functions for computing metrics for uncertainty quantification.

- :ref:`Visualization Module <tfmelt.utils.visualization>`: Functions for visualizing
  data. Includes functions for plotting data, model predictions, and uncertainty
  metrics.


.. _tfmelt.utils.evaluation:

Evaluation Module
-----------------

.. automodule:: tfmelt.utils.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

.. _tfmelt.utils.preprocessing:

Preprocessing Module
--------------------

.. automodule:: tfmelt.utils.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

.. _tfmelt.utils.statistics:

Statistics Module
-----------------

.. automodule:: tfmelt.utils.statistics
   :members:
   :undoc-members:
   :show-inheritance:

.. _tfmelt.utils.visualization:

Visualization Module
--------------------

.. autofunction:: tfmelt.utils.visualization.plot_interval_width_vs_value

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import plot_interval_width_vs_value

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 3
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # Plot interval width vs. value
  fig, ax = plt.subplots()
  plot_interval_width_vs_value(y_true, y_pred, y_std, ax, "Linear Regression")
  plt.show()


.. autofunction:: tfmelt.utils.visualization.plot_qq

.. plot::
  
  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import plot_qq

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 3
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # Plot Q-Q plot
  fig, ax = plt.subplots()
  plot_qq(ax, y_true, y_pred, y_std, "Linear Regression")
  plt.show()

.. autofunction:: tfmelt.utils.visualization.plot_residuals_vs_value

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import plot_residuals_vs_value

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 3
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # Plot residuals vs. value
  fig, ax = plt.subplots()
  plot_residuals_vs_value(y_true, y_pred, ax, "Linear Regression")
  plt.show()

.. autofunction:: tfmelt.utils.visualization.plot_uncertainty_calibration

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import plot_uncertainty_calibration

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 3
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # Plot uncertainty calibration
  fig, ax = plt.subplots()
  plot_uncertainty_calibration(ax, y_true, y_pred, y_std, "Linear Regression")
  plt.show()


.. autofunction:: tfmelt.utils.visualization.plot_uncertainty_distribution

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import plot_uncertainty_distribution

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 3
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # Plot uncertainty distribution
  fig, ax = plt.subplots()
  plot_uncertainty_distribution(y_std, ax, "Linear Regression")
  plt.show()


.. autofunction:: tfmelt.utils.visualization.point_cloud_plot

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import point_cloud_plot
  from tfmelt.utils.statistics import compute_rsquared, compute_rmse

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 1
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # compute metrics
  r_squared = compute_rsquared(y_true, y_pred)
  rmse = compute_rmse(y_true, y_pred)

  # Plot point cloud
  fig, ax = plt.subplots()
  point_cloud_plot(ax, y_true, y_pred, r_squared, rmse, "Linear Regression")
  plt.show()


.. autofunction:: tfmelt.utils.visualization.point_cloud_plot_with_uncertainty

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from tfmelt.utils.visualization import point_cloud_plot_with_uncertainty
  from tfmelt.utils.statistics import compute_rsquared, compute_rmse

  np.random.seed(42)

  # Generate data for a simple linear regression problem
  num_samples = 1000
  num_outputs = 1
  x = np.linspace(0, 10, num_samples).reshape(-1, 1)
  y_true = 3 * x + 7 + np.random.normal(0, 1, (num_samples, num_outputs))

  # Simulate predictions with normally distributed errors
  y_pred = y_true + np.random.normal(0, 0.1, (num_samples, num_outputs))

  # Simulate predicted standard deviations as a quadratic function of x
  coefficients = np.random.uniform(0.1, 0.3, num_outputs).reshape(1, num_outputs)
  y_std = coefficients + 0.01 * (x - 5)**2

  # compute metrics
  r_squared = compute_rsquared(y_true, y_pred)
  rmse = compute_rmse(y_true, y_pred)

  # Plot point cloud with uncertainty
  fig, ax = plt.subplots()
  point_cloud_plot_with_uncertainty(ax, y_true, y_pred, y_std)
  plt.show()



.. automodule:: tfmelt.utils.visualization
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: plot_interval_width_vs_value, plot_qq, plot_residuals_vs_value,
                     plot_uncertainty_calibration, plot_uncertainty_distribution,
                     point_cloud_plot, point_cloud_plot_with_uncertainty


