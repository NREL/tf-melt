from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot

from .statistics import compute_metrics, compute_rmse, compute_rsquared


def plot_history(
    history,
    metrics: Optional[List] = ["loss"],
    plot_log: Optional[bool] = False,
    savename: Optional[str] = None,
):
    """
    Plot training history for specified metrics and optionally save the plot.

    Args:
        history: History object from model training.
        metrics (list of str): List of metrics to plot. Defaults to ["loss"].
        plot_log (bool): Whether to include a logarithmic scale subplot. Defaults to
                         False.
        savename (str): Full path to save the plot image. If None, the plot will not be
                        saved. Defaults to None.
    """
    # TODO: return the figure object for further customization
    # If plot_log is True, create a 1x2 subplot with normal and log scales
    if plot_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = None

    # Plot metrics for both training and validation sets
    for metric in metrics:
        ax1.plot(history.history[metric], label=f"train {metric}")
        if f"val_{metric}" in history.history:
            ax1.plot(history.history[f"val_{metric}"], label=f"validation {metric}")

        if plot_log:
            ax2.plot(history.history[metric], label=f"train {metric}")
            if f"val_{metric}" in history.history:
                ax2.plot(history.history[f"val_{metric}"], label=f"validation {metric}")

    # Set plot labels and legend
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

    # Save the plot if a filename is provided, otherwise display it
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


def point_cloud_plot(
    ax,
    y_real,
    y_pred,
    r_squared,
    rmse,
    label: Optional[str] = None,
    marker: Optional[str] = "o",
    color: Optional[str] = "blue",
    text_pos: Optional[tuple] = (0.3, 0.01),
):
    """
    Create a point cloud plot on the given axes.

    Args:
        ax: Matplotlib axes object.
        y_real (array-like): Actual values.
        y_pred (array-like): Predicted values.
        r_squared (float): R-squared value.
        rmse (float): RMSE value.
        label (str, optional): Label for the plot. Defaults to None.
        marker (str, optional): Marker style. Defaults to "o".
        color (str, optional): Marker color. Defaults to "blue".
        text_pos (tuple, optional): Position for the RMSE text annotation (x, y).
                                    Defaults to (0.3, 0.01).
    """
    # Plot the point cloud
    ax.plot(y_real, y_pred, marker=marker, linestyle="None", label=label, color=color)
    ax.plot(y_real, y_real, linestyle="dashed", color="grey")
    # Add text annotation for R-squared and RMSE
    # TODO: Add more metrics to the text annotation similar to the UQ plot
    # TODO: Add ability to change the formatting of the text annotation
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
    output_indices: Optional[List[int]] = None,
    max_targets: Optional[int] = 3,
    savename: Optional[str] = None,
):
    """
    Plot predictions for specified output indices.

    Args:
        pred_train (array-like): Predicted training values.
        y_train_real (array-like): Actual training values.
        pred_val (array-like): Predicted validation values.
        y_val_real (array-like): Actual validation values.
        pred_test (array-like): Predicted test values.
        y_test_real (array-like): Actual test values.
        output_indices (list of int, optional): List of output indices to plot.
                                                Defaults to None.
        max_targets (int, optional): Maximum number of targets to plot. Defaults to 3.
        savename (str, optional): Full path to save the plot image. If None, the plot
                                  will not be saved. Defaults to None.
    """
    # If output_indices is None, plot the first max_targets outputs
    if output_indices is None:
        output_indices = list(range(min(max_targets, pred_train.shape[1])))

    # Create a 1x3 subplot for training, validation, and test data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define markers and colors for the point cloud plot
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    colors = plt.cm.tab10.colors

    # Define text positions for the metrics text annotation
    text_positions = [(0.3, i * 0.05 + 0.01) for i in range(len(output_indices))]

    # Plot predictions for each output index
    for i, idx in enumerate(output_indices):
        # Compute R-squared and RMSE for each dataset
        r_sq_train = compute_rsquared(y_train_real[:, idx], pred_train[:, idx])
        rmse_train = compute_rmse(y_train_real[:, idx], pred_train[:, idx])
        r_sq_val = compute_rsquared(y_val_real[:, idx], pred_val[:, idx])
        rmse_val = compute_rmse(y_val_real[:, idx], pred_val[:, idx])
        r_sq_test = compute_rsquared(y_test_real[:, idx], pred_test[:, idx])
        rmse_test = compute_rmse(y_test_real[:, idx], pred_test[:, idx])

        # Create point cloud plot for each dataset
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

    # Set plot titles
    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot if a filename is provided, otherwise display it
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


def point_cloud_plot_with_uncertainty(
    ax,
    y_real,
    y_pred,
    y_std,
    text_pos: Optional[tuple] = (0.05, 0.95),
    metrics_to_display: Optional[List[str]] = None,
):
    """
    Create a point cloud plot with uncertainty on the given axes.

    Args:
        ax: Matplotlib axes object.
        y_real (array-like): Actual values.
        y_pred (array-like): Predicted values.
        y_std (array-like): Standard deviation of predictions.
        text_pos (tuple, optional): Position for the text annotation (x, y). Defaults to
                                    (0.05, 0.95).
        metrics_to_display (list of str, optional): List of metrics to display in the
                                                    text annotation. If None, all
                                                    metrics in compute_metrics() are
                                                    show. Defaults to None.
    """
    # TODO: Make the metrics_to_display argument more straightforward
    cmap = plt.get_cmap("viridis")
    # TODO: Add in option to normalize the standard deviation predictions
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

    # Add text annotation for metrics
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
    metrics_to_display: Optional[List[str]] = None,
    savename: Optional[str] = None,
):
    """
    Plot predictions with uncertainty for training, validation, and test data.

    Args:
        mean_train, std_train, y_train_real (array-like): Training data.
        mean_val, std_val, y_val_real (array-like): Validation data.
        mean_test, std_test, y_test_real (array-like): Test data.
        metrics_to_display (list of str, optional): List of metrics to display in the
                                                    text annotation. If None, all
                                                    metrics in compute_metrics() are
                                                    show. Defaults to None.
        savename (str, optional): Full path to save the plot image. If None, the plot
                                  will not be saved. Defaults to None.
    """
    # Create a 1x3 subplot for training, validation, and test data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot predictions with uncertainty for each dataset
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

    # Set plot titles
    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions with Uncertainty")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot if a filename is provided, otherwise display it
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


def plot_uncertainty_distribution(
    y_std,
    ax,
    dataset_name,
    colors: Optional[List] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
):
    """
    Plot the distribution of uncertainty (standard deviation) values.

    Args:
        y_std (array-like): Standard deviation of predictions.
        ax: Matplotlib axes object.
        dataset_name (str): Name of the dataset for labeling.
        colors (list, optional): List of colors for the plot. Defaults to the default
                                 color cycle.
    """
    n_outputs = y_std.shape[1]
    handles = []
    # Plot the histogram of standard deviation values for each output
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


def plot_residuals_vs_value(
    y_true,
    y_pred,
    ax,
    dataset_name,
    colors: Optional[List] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    use_pred: Optional[bool] = True,
):
    """
    Plot the residuals (true - predicted) against the true or predicted values.

    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        ax: Matplotlib axes object.
        dataset_name (str): Name of the dataset for labeling.
        colors (list, optional): List of colors for the plot. Defaults to the default
                                 color cycle.
        use_pred (bool, optional): Whether to use predicted or true values on the
                                   x-axis. Defaults to True.
    """
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
        f"{dataset_name} Data: Residuals vs. {'Predicted' if use_pred else 'True'} "
        f"Values"
    )
    ax.set_xlabel("Predicted Values" if use_pred else "True Values")
    ax.set_ylabel("Residual (True - Predicted)")
    ax.legend(handles=handles)


def plot_interval_width_vs_value(
    y_true,
    y_pred,
    y_std,
    ax,
    dataset_name,
    colors: Optional[List] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    normalize: Optional[bool] = False,
    use_pred: Optional[bool] = False,
):
    """
    Plot the prediction interval width against the true or predicted values.

    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        y_std (array-like): Standard deviation of predictions.
        ax: Matplotlib axes object.
        dataset_name (str): Name of the dataset for labeling.
        colors (list, optional): List of colors for the plot. Defaults to the default
                                 color cycle.
        normalize (bool, optional): Whether to normalize the interval width by the range
                                    of the true values. Defaults to False.
        use_pred (bool, optional): Whether to use predicted or true values on the
                                   x-axis. Defaults to False.
    """
    n_outputs = y_std.shape[1]
    handles = []
    for i in range(n_outputs):
        # Compute the interval width as 2 * 1.96 * standard deviation (95% CI)
        interval_width = 2 * 1.96 * y_std[:, i]

        # Normalize the interval width by the range of the true values
        if normalize:
            interval_width /= np.max(y_true[:, i]) - np.min(y_true[:, i])

        h = ax.scatter(
            y_pred[:, i] if use_pred else y_true[:, i],
            interval_width,
            alpha=0.5,
            edgecolor="k",
            linewidth=0.5,
            color=colors[i % len(colors)],
            label=f"Output {i+1}",
        )
        handles.append(h)  # Get a handle to the scatter for the legend
    ax.set_title(
        f"{dataset_name} Data: Interval Width vs. {'Predicted' if use_pred else 'True'}"
        f" Values"
    )
    ax.set_xlabel("Predicted Values" if use_pred else "True Values")
    if normalize:
        ax.set_ylabel("Normalized Prediction Interval Width")
    else:
        ax.set_ylabel("Prediction Interval Width")
    ax.legend(handles=handles)


def plot_qq(
    ax,
    y_true,
    y_pred,
    y_std,
    dataset_name,
    colors: Optional[List] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
):
    """
    Plot Quantile-Quantile (Q-Q) plot using scipy.stats.probplot.

    Args:
        ax: Matplotlib axes object.
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        y_std (array-like): Standard deviation of predictions.
        dataset_name (str): Name of the dataset for labeling.
        colors (list, optional): List of colors for the plot.
    """
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    handles = []
    for i in range(n_outputs):
        normalized_residuals = (y_true[:, i] - y_pred[:, i]) / y_std[:, i]
        probplot(normalized_residuals, dist="norm", plot=ax, fit=True, rvalue=False)
        # Get the handle for the QQ plot line and style it
        h = ax.get_lines()[-2]
        h.set_color(colors[i % len(colors)])
        handles.append(h)
        # Get the handle of the reference line and style it
        h = ax.get_lines()[-1]
        h.set_linestyle("--")
        h.set_color(colors[i % len(colors)])
        h.set_alpha(0.5)
    # Set plot labels and title
    ax.set_title(f"Q-Q Plot: {dataset_name}")
    ax.legend(handles=handles, labels=[f"Output {i+1}" for i in range(n_outputs)])


def plot_uncertainty_calibration(
    ax,
    y_true,
    y_pred,
    y_std,
    dataset_name,
    colors: Optional[List] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
):
    """
    Plot Uncertainty Calibration. This plot shows the absolute prediction error against
    the predicted uncertainty. The dashed line represents perfect calibration.

    Args:
        ax: Matplotlib axes object.
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        y_std (array-like): Standard deviation of predictions.
        dataset_name (str): Name of the dataset for labeling.
        colors (list): List of colors for the plot.
    """
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    handles = []
    # Plot the absolute prediction error against the predicted uncertainty
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


# def plot_coverage_by_quantile(
#     ax,
#     y_true,
#     y_pred,
#     y_std,
#     dataset_name,
#     colors: Optional[List] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
# ):
#     """
#     Plot Coverage Probability by quantile.

#     Args:
#         ax: Matplotlib axes object.
#         y_true (array-like): Actual values.
#         y_pred (array-like): Predicted values.
#         y_std (array-like): Standard deviation of predictions.
#         dataset_name (str): Name of the dataset for labeling.
#         colors (list): List of colors for the plot.
#     """
#     # TODO: Make this more informative...
#     n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
#     handles = []
#     quantiles = np.percentile(y_std, np.linspace(0, 100, 100))
#     text_positions = [(0.8, 0.2 - i * 0.05) for i in range(n_outputs)]

#     for i in range(n_outputs):
#         coverages = []
#         for q in quantiles:
#             in_interval = np.abs(y_true[:, i] - y_pred[:, i]) <= q
#             coverage = np.mean(in_interval)
#             coverages.append(coverage)

#         # Find the quantile that achieves 95% coverage
#         coverage_array = np.array(coverages)
#         quantile_95 = quantiles[np.searchsorted(coverage_array, 0.95)]

#         h = ax.plot(
#             quantiles, coverages, label=f"Output {i+1}", color=colors[i % len(colors)]
#         )
#         handles.append(h[0])

#         # Add vertical line at the quantile for 95% coverage
#         ax.axvline(x=quantile_95, color=colors[i % len(colors)], linestyle="--")

#         # Add text box showing the quantile value
#         ax.text(
#             quantile_95,
#             0.95,
#             f"95% Coverage at {quantile_95:.2f}",
#             transform=ax.transAxes,
#             verticalalignment="bottom",
#             horizontalalignment="right",
#             backgroundcolor="white",
#             color=colors[i % len(colors)],
#             position=text_positions[i],
#         )

#     ax.set_xlabel("Uncertainty Quantile")
#     ax.set_ylabel("Coverage Probability")
#     ax.set_title(f"Coverage by Quantile: {dataset_name}")
#     ax.legend(handles=handles, loc="center right")
