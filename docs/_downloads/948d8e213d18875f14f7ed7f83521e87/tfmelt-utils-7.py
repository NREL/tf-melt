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