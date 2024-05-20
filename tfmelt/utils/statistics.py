import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import mean_squared_error


def compute_rsquared(truth, pred):
    """Compute the coefficient of determination (R^2)."""
    rss = np.sum((pred - truth) ** 2)
    tss = np.sum((truth - np.mean(truth)) ** 2)
    r_sq = 1 - rss / tss
    return r_sq


def compute_rmse(truth, pred):
    """Compute the root mean squared error (RMSE)."""
    return np.sqrt(mean_squared_error(truth, pred))


def compute_normalized_rmse(truth, pred):
    """Compute the normalized root mean squared error (NRMSE)."""
    rmse = compute_rmse(truth, pred)
    return rmse / (truth.max() - truth.min())


def compute_picp(truth, mean, std):
    """Compute the prediction interval coverage probability (PICP)."""
    lower_bound = mean - 1.96 * std
    upper_bound = mean + 1.96 * std
    coverage = np.mean((truth >= lower_bound) & (truth <= upper_bound))
    return coverage


def compute_mpiw(std):
    """Compute the mean prediction interval width (MPIW)."""
    return np.mean(2 * 1.96 * std)


def compute_expected_calibration_error(truth, mean, std, n_bins=10):
    """Compute the expected calibration error (ECE)."""
    confidences = norm.cdf((truth - mean) / std)
    sorted_indices = np.argsort(confidences)
    bin_size = len(confidences) // n_bins

    ece = 0.0
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size if i < n_bins - 1 else len(confidences)
        bin_indices = sorted_indices[bin_start:bin_end]
        bin_confidence = np.mean(confidences[bin_indices])
        bin_accuracy = np.mean(
            (truth[bin_indices] >= mean[bin_indices] - 1.96 * std[bin_indices])
            & (truth[bin_indices] <= mean[bin_indices] + 1.96 * std[bin_indices])
        )
        ece += np.abs(bin_confidence - bin_accuracy) * len(bin_indices) / len(truth)

    return ece


def pdf_based_nll(y_true, y_pred, std):
    """
    Compute the PDF-based negative log-likelihood (NLL).

    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted means.
    std (array-like): Predicted standard deviations.

    Returns:
    float: PDF-based negative log-likelihood.
    """
    return -np.mean(norm.logpdf(y_true, y_pred, std))


def compute_crps(truth, mean, std):
    """Compute the continuous ranked probability score (CRPS)."""
    crps = np.mean(
        [
            norm.cdf(t, loc=m, scale=s) * (2 * norm.cdf(t, loc=m, scale=s) - 2)
            for t, m, s in zip(truth, mean, std)
        ]
    )
    return crps


def compute_sharpness(std):
    """Compute the sharpness of the predictive distribution."""
    return np.mean(std)


def compute_cwc(truth, mean, std, alpha=0.95, gamma=1.0):
    """Compute the converage width criterion (CWC)."""
    picp = compute_picp(truth, mean, std)
    mpiw = compute_mpiw(std)
    penalty = gamma * max(0, alpha - picp)
    cwc = mpiw * (1 + penalty)
    return cwc


def compute_pinaw(truth, mean, std):
    """Compute the prediction interval normalized average width (PINAW)."""
    mpiw = compute_mpiw(std)
    return mpiw / (truth.max() - truth.min())


def compute_winkler_score(truth, mean, std, alpha=0.05):
    """Compute the Winkler score."""
    lower_bound = mean - norm.ppf(1 - alpha / 2) * std
    upper_bound = mean + norm.ppf(1 - alpha / 2) * std
    score = np.where(
        truth < lower_bound,
        upper_bound - lower_bound + 2 / alpha * (lower_bound - truth),
        np.where(
            truth > upper_bound,
            upper_bound - lower_bound + 2 / alpha * (truth - upper_bound),
            upper_bound - lower_bound,
        ),
    )
    return np.mean(score)


def compute_brier_score(truth, prob):
    """Compute the Brier score."""
    return np.mean((prob - truth) ** 2)


def cdf_based_nll(y_true, y_pred, std):
    """
    Compute the CDF-based negative log-likelihood (NLL).

    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted means.
    std (array-like): Predicted standard deviations.

    Returns:
    float: CDF-based negative log-likelihood.
    """
    confidences = norm.cdf((y_true - y_pred) / std)
    return -np.mean(np.log(confidences))


def fit_temperature(y_true, y_pred, std):
    initial_temperature = 1.0
    result = minimize(
        # lambda t: nll(y_true, y_pred, std * t),
        lambda t: cdf_based_nll(y_true, y_pred, std * t),
        initial_temperature,
        bounds=[(0.1, 10.0)],
    )
    return result.x


def calibrate_uncertainty(std, temperature):
    return std * temperature


def compute_metrics(y_real, y_pred, y_std, metrics_to_compute=None):
    """
    Compute various metrics between real and predicted values.

    Parameters:
    y_real (array-like): Actual values.
    y_pred (array-like): Predicted values.
    y_std (array-like): Standard deviation of predictions.
    metrics_to_compute (list, optional): List of metrics to compute. If None, all metrics are computed.

    Returns:
    dict: Dictionary of computed metrics.
    """
    all_metrics = {
        "R^2": (compute_rsquared, (y_real, y_pred)),
        "RMSE": (compute_rmse, (y_real, y_pred)),
        "NRMSE": (compute_normalized_rmse, (y_real, y_pred)),
        "PICP": (compute_picp, (y_real, y_pred, y_std)),
        "MPIW": (compute_mpiw, (y_std,)),
        "ECE": (compute_expected_calibration_error, (y_real, y_pred, y_std)),
        "NLL": (pdf_based_nll, (y_real, y_pred, y_std)),
        "CRPS": (compute_crps, (y_real, y_pred, y_std)),
        "Sharpness": (compute_sharpness, (y_std,)),
        "CWC": (compute_cwc, (y_real, y_pred, y_std)),
        "PINAW": (compute_pinaw, (y_real, y_pred, y_std)),
        "Winkler Score": (compute_winkler_score, (y_real, y_pred, y_std)),
    }

    if metrics_to_compute is None:
        metrics_to_compute = all_metrics.keys()

    metrics = {}
    for metric in metrics_to_compute:
        if metric in all_metrics:
            func, args = all_metrics[metric]
            metrics[metric] = func(*args)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics
