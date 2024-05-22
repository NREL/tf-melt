import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot

from .statistics import compute_metrics, compute_rmse, compute_rsquared


def plot_history(history, metrics=["loss"], plot_log=False, savename=None):
    """
    Plot training history for specified metrics and optionally save the plot.

    Parameters:
    history: History object from model training.
    metrics (list of str): List of metrics to plot. Defaults to ["loss"].
    plot_log (bool): Whether to include a logarithmic scale subplot. Defaults to False.
    savename (str): Full path to save the plot image. If None, the plot will not be
                    saved. Defaults to None.

    Returns:
    None
    """
    # TODO: return the figure object for further customization
    if plot_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = None

    for metric in metrics:
        ax1.plot(history.history[metric], label=f"train {metric}")
        if f"val_{metric}" in history.history:
            ax1.plot(history.history[f"val_{metric}"], label=f"validation {metric}")

        if plot_log:
            ax2.plot(history.history[metric], label=f"train {metric}")
            if f"val_{metric}" in history.history:
                ax2.plot(history.history[f"val_{metric}"], label=f"validation {metric}")

    ax1.legend()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Metrics")

    if plot_log:
        ax2.legend()
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Metrics")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    fig.tight_layout()

    if savename:
        fig.savefig(savename)
    else:
        plt.show()


def point_cloud_plot(
    ax, y_real, y_pred, r_squared, rmse, label, marker, color, text_pos=(0.3, 0.01)
):
    """
    Create a point cloud plot on the given axes.

    Parameters:
    ax: Matplotlib axes object.
    y_real (array-like): Actual values.
    y_pred (array-like): Predicted values.
    r_squared (float): R-squared value.
    rmse (float): RMSE value.
    label (str): Label for the plot.
    marker (str): Marker style.
    color (str): Marker color.
    text_pos (tuple): Position for the RMSE text annotation (x, y).

    Returns:
    None
    """
    ax.plot(y_real, y_pred, marker=marker, linestyle="None", label=label, color=color)
    ax.plot(y_real, y_real, linestyle="dashed", color="grey")
    ax.text(
        *text_pos,
        rf"R$^2$ = {r_squared:0.3f}, RMSE = {rmse:0.3f}",
        transform=ax.transAxes,
        color=color,
    )
    ax.legend()
    ax.set_xlabel("truth")
    ax.set_ylabel("prediction")


def plot_predictions(
    pred_train,
    y_train_real,
    pred_val,
    y_val_real,
    pred_test,
    y_test_real,
    output_indices=None,
    max_targets=3,
    savename=None,
):
    """
    Plot predictions for specified output indices.

    Parameters:
    pred_train (array-like): Predicted training values.
    y_train_real (array-like): Actual training values.
    pred_val (array-like): Predicted validation values.
    y_val_real (array-like): Actual validation values.
    pred_test (array-like): Predicted test values.
    y_test_real (array-like): Actual test values.
    output_indices (list of int): List of output indices to plot. Defaults to None.
    max_targets (int): Maximum number of targets to plot. Defaults to 3.
    savename (str): Full path to save the plot image. If None, the plot will not be
                    saved. Defaults to None.

    Returns:
    None
    """
    if output_indices is None:
        output_indices = list(range(min(max_targets, pred_train.shape[1])))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    colors = plt.cm.tab10.colors

    text_positions = [(0.3, i * 0.05 + 0.01) for i in range(len(output_indices))]

    for i, idx in enumerate(output_indices):
        r_sq_train = compute_rsquared(y_train_real[:, idx], pred_train[:, idx])
        rmse_train = compute_rmse(y_train_real[:, idx], pred_train[:, idx])
        r_sq_val = compute_rsquared(y_val_real[:, idx], pred_val[:, idx])
        rmse_val = compute_rmse(y_val_real[:, idx], pred_val[:, idx])
        r_sq_test = compute_rsquared(y_test_real[:, idx], pred_test[:, idx])
        rmse_test = compute_rmse(y_test_real[:, idx], pred_test[:, idx])

        point_cloud_plot(
            axes[0],
            y_train_real[:, idx],
            pred_train[:, idx],
            r_sq_train,
            rmse_train,
            f"Output {idx}",
            markers[i % len(markers)],
            colors[i % len(colors)],
            text_pos=text_positions[i % len(text_positions)],
        )
        point_cloud_plot(
            axes[1],
            y_val_real[:, idx],
            pred_val[:, idx],
            r_sq_val,
            rmse_val,
            f"Output {idx}",
            markers[i % len(markers)],
            colors[i % len(colors)],
            text_pos=text_positions[i % len(text_positions)],
        )
        point_cloud_plot(
            axes[2],
            y_test_real[:, idx],
            pred_test[:, idx],
            r_sq_test,
            rmse_test,
            f"Output {idx}",
            markers[i % len(markers)],
            colors[i % len(colors)],
            text_pos=text_positions[i % len(text_positions)],
        )

    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savename:
        fig.savefig(savename)
    else:
        plt.show()


def point_cloud_plot_with_uncertainty(
    ax,
    y_real,
    y_pred,
    y_std,
    label,
    text_pos=(0.05, 0.95),
    metrics_to_display=None,
):
    """
    Create a point cloud plot with uncertainty on the given axes.

    Parameters:
    ax: Matplotlib axes object.
    y_real (array-like): Actual values.
    y_pred (array-like): Predicted values.
    y_std (array-like): Standard deviation of predictions.
    label (str): Label for the plot.
    text_pos (tuple): Position for the text annotation (x, y).

    Returns:
    None
    """
    cmap = plt.get_cmap("viridis")
    # pcnorm = plt.Normalize(y_std.min(), y_std.max())
    sc = ax.scatter(
        y_real,
        y_pred,
        c=y_std,
        cmap=cmap,
        # norm=pcnorm,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )

    # Plot perfect prediction line
    min_val = min(np.min(y_real), np.min(y_pred))
    max_val = max(np.max(y_real), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="dashed", color="grey")

    # Compute metrics
    metrics = compute_metrics(
        y_real, y_pred, y_std, metrics_to_compute=metrics_to_display
    )
    textstr = "\n".join([f"{key} = {value:.3f}" for key, value in metrics.items()])

    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    ax.text(
        *text_pos,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Uncertainty (std dev)")


def plot_predictions_with_uncertainty(
    mean_train,
    std_train,
    y_train_real,
    mean_val,
    std_val,
    y_val_real,
    mean_test,
    std_test,
    y_test_real,
    metrics_to_display=None,
):
    """
    Plot predictions with uncertainty for training, validation, and test data.

    Parameters:
    mean_train, std_train, y_train_real (array-like): Training data.
    mean_val, std_val, y_val_real (array-like): Validation data.
    mean_test, std_test, y_test_real (array-like): Test data.

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = {
        "Train": (mean_train, std_train, y_train_real, axes[0]),
        "Validation": (mean_val, std_val, y_val_real, axes[1]),
        "Test": (mean_test, std_test, y_test_real, axes[2]),
    }

    for dataset_name, (mean, std, y_real, ax) in datasets.items():
        point_cloud_plot_with_uncertainty(
            ax,
            y_real,
            mean,
            std,
            f"{dataset_name} Data",
            metrics_to_display=metrics_to_display,
        )

    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions with Uncertainty")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_uncertainty_distribution(y_std, ax, dataset_name, colors):
    n_outputs = y_std.shape[1]
    handles = []
    for i in range(n_outputs):
        h = ax.hist(
            y_std[:, i],
            bins=30,
            alpha=0.5,
            edgecolor="black",
            color=colors[i % len(colors)],
            label=f"Output {i+1}",
        )
        handles.append(h[2][0])  # Get a handle to the patch for the legend
    ax.set_title(f"{dataset_name} Data: Uncertainty Distribution")
    ax.set_xlabel("Standard Deviation (Uncertainty)")
    ax.set_ylabel("Frequency")
    ax.legend(handles=handles)


def plot_value_vs_residuals(y_true, y_pred, ax, dataset_name, colors, use_pred=True):
    n_outputs = y_true.shape[1]
    handles = []
    for i in range(n_outputs):
        residuals = y_true[:, i] - y_pred[:, i]
        h = ax.scatter(
            y_pred[:, i] if use_pred else y_true[:, i],
            residuals,
            alpha=0.5,
            edgecolor="k",
            linewidth=0.5,
            color=colors[i % len(colors)],
            label=f"Output {i+1}",
        )
        handles.append(h)  # Get a handle to the scatter for the legend
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(
        f"{dataset_name} Data: {'Predicted' if use_pred else 'True'} Value vs. "
        f"Residuals"
    )
    ax.set_xlabel("Predicted Value" if use_pred else "True Value")
    ax.set_ylabel("Residual (True - Predicted)")
    ax.legend(handles=handles)


def plot_interval_width_vs_true_values(
    y_true, y_pred, y_std, ax, dataset_name, colors, normalize=False
):
    n_outputs = y_std.shape[1]
    handles = []
    for i in range(n_outputs):
        interval_width = 2 * 1.96 * y_std[:, i]
        if normalize:
            interval_width /= np.max(y_true[:, i])
        h = ax.scatter(
            y_true[:, i],
            interval_width,
            alpha=0.5,
            edgecolor="k",
            linewidth=0.5,
            color=colors[i % len(colors)],
            label=f"Output {i+1}",
        )
        handles.append(h)  # Get a handle to the scatter for the legend
    ax.set_title(f"{dataset_name} Data: Interval Width vs. True Values")
    ax.set_xlabel("True Value")
    if normalize:
        ax.set_ylabel("Normalized Prediction Interval Width")
    else:
        ax.set_ylabel("Prediction Interval Width")
    ax.legend(handles=handles)


def plot_qq(ax, y_true, y_pred, y_std, dataset_name, colors):
    """
    Plot Quantile-Quantile (Q-Q) Plot.

    Parameters:
    ax: Matplotlib axes object.
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.
    y_std (array-like): Standard deviation of predictions.
    dataset_name (str): Name of the dataset for labeling.
    colors (list): List of colors for the plot.

    Returns:
    None
    """
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    handles = []
    for i in range(n_outputs):
        normalized_residuals = (y_true[:, i] - y_pred[:, i]) / y_std[:, i]
        probplot(normalized_residuals, dist="norm", plot=ax)
        h = ax.get_lines()[-2]  # Get the handle for the QQ plot line
        h.set_color(colors[i % len(colors)])
        handles.append(h)
        # Get the handle of the reference line
        h = ax.get_lines()[-1]
        h.set_linestyle("--")
        h.set_color(colors[i % len(colors)])
        h.set_alpha(0.5)
    ax.set_title(f"Q-Q Plot: {dataset_name}")
    ax.legend(handles=handles, labels=[f"Output {i+1}" for i in range(n_outputs)])


def plot_uncertainty_calibration(ax, y_true, y_pred, y_std, dataset_name, colors):
    """
    Plot Uncertainty Calibration.

    Parameters:
    ax: Matplotlib axes object.
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.
    y_std (array-like): Standard deviation of predictions.
    dataset_name (str): Name of the dataset for labeling.
    colors (list): List of colors for the plot.

    Returns:
    None
    """
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    handles = []
    for i in range(n_outputs):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        h = ax.scatter(
            y_std[:, i],
            errors,
            alpha=0.5,
            edgecolor="k",
            linewidth=0.5,
            color=colors[i % len(colors)],
            label=f"Output {i+1}",
        )
        handles.append(h)
    max_std = np.max(y_std)
    ax.plot([0, max_std], [0, max_std], linestyle="--", color="grey")
    ax.set_xlabel("Predicted Uncertainty (std dev)")
    ax.set_ylabel("Absolute Prediction Error")
    ax.set_title(f"Uncertainty Calibration: {dataset_name}")

    # Adding text annotations for overconfident and underconfident regions
    ax.text(
        0.8,
        0.2,
        "Underconfident",
        color="red",
        fontsize=12,
        ha="center",
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.2,
        0.8,
        "Overconfident",
        color="blue",
        fontsize=12,
        ha="center",
        weight="bold",
        transform=ax.transAxes,
    )

    ax.legend(handles=handles)


def plot_coverage_by_quantile(ax, y_true, y_pred, y_std, dataset_name, colors):
    """
    Plot Coverage Probability by Quantile.

    Parameters:
    ax: Matplotlib axes object.
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.
    y_std (array-like): Standard deviation of predictions.
    dataset_name (str): Name of the dataset for labeling.
    colors (list): List of colors for the plot.

    Returns:
    None
    """
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    handles = []
    quantiles = np.percentile(y_std, np.linspace(0, 100, 100))
    text_positions = [(0.8, 0.2 - i * 0.05) for i in range(n_outputs)]

    for i in range(n_outputs):
        coverages = []
        for q in quantiles:
            in_interval = np.abs(y_true[:, i] - y_pred[:, i]) <= q
            coverage = np.mean(in_interval)
            coverages.append(coverage)

        # Find the quantile that achieves 95% coverage
        coverage_array = np.array(coverages)
        quantile_95 = quantiles[np.searchsorted(coverage_array, 0.95)]

        h = ax.plot(
            quantiles, coverages, label=f"Output {i+1}", color=colors[i % len(colors)]
        )
        handles.append(h[0])

        # Add vertical line at the quantile for 95% coverage
        ax.axvline(x=quantile_95, color=colors[i % len(colors)], linestyle="--")

        # Add text box showing the quantile value
        ax.text(
            quantile_95,
            0.95,
            f"95% Coverage at {quantile_95:.2f}",
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            backgroundcolor="white",
            color=colors[i % len(colors)],
            position=text_positions[i],
        )

    ax.set_xlabel("Uncertainty Quantile")
    ax.set_ylabel("Coverage Probability")
    ax.set_title(f"Coverage by Quantile: {dataset_name}")
    ax.legend(handles=handles, loc="center right")
