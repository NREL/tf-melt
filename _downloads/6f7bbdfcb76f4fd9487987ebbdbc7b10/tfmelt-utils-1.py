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