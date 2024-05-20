import warnings

import numpy as np


def make_predictions(model, x_data, y_normalizer=None, unnormalize=False):
    """
    Make predictions using the provided model and optionally unscale the results.

    Parameters:
    model: The trained model to use for making predictions.
    x_data (array-like): Data features to make predictions on.
    y_normalizer: The normalizer used for the target variable, to unscale the predictions (optional).
    unnormalize (bool): Whether to unscale the predictions using the y_normalizer. Defaults to False.

    Returns:
    array-like: Predictions (unnormalized if specified).
    """
    # Make predictions
    predictions = model(x_data, training=False)

    # Unscale the results if required
    if unnormalize and y_normalizer is not None:
        predictions = y_normalizer.inverse_transform(predictions)
    elif unnormalize and y_normalizer is None:
        raise ValueError(
            "y_normalizer must be provided to unnormalize the predictions."
        )
    elif not unnormalize and y_normalizer is not None:
        warnings.warn(
            "y_normalizer is provided but unnormalize is set to False. "
            "Set unnormalize to True to unnormalize the predictions."
        )

    return predictions


def ensemble_predictions(
    model, x_data, y_normalizer=None, unnormalize=False, n_iter=100, training=False
):
    """
    Make ensemble predictions using the provided model and optionally unscale the results.

    Parameters:
    model: The trained model to use for making predictions.
    x_data (array-like): Data features to make predictions on.
    y_normalizer: The normalizer used for the target variable, to unscale the predictions (optional).
    unnormalize (bool): Whether to unscale the predictions using the y_normalizer. Defaults to False.
    n_iter (int): Number of iterations for ensemble predictions. Defaults to 100.
    training (bool): Whether to use training mode for making predictions. Defaults to False.

    Returns:
    tuple: Mean and standard deviation of predictions (unnormalized if specified).
    """
    predictions = []
    for _ in range(n_iter):
        pred = model(x_data, training=training)
        pred_np = pred.numpy()
        if unnormalize and y_normalizer is not None:
            pred_unscaled = y_normalizer.inverse_transform(pred_np)
        elif unnormalize and y_normalizer is None:
            raise ValueError(
                "y_normalizer must be provided to unnormalize the predictions."
            )
        elif not unnormalize and y_normalizer is not None:
            warnings.warn(
                "y_normalizer is provided but unnormalize is set to False. "
                "Set unnormalize to True to unnormalize the predictions."
            )
        else:
            pred_unscaled = pred_np
        predictions.append(pred_unscaled)

    predictions = np.array(predictions)
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    return pred_mean, pred_std


def extract_statistics_from_pdf(model, x_data, y_normalizer=None, unnormalize=False):
    """
    Extract statistics from the model's PDF predictions and optionally unscale the results.

    Parameters:
    model: The trained model to use for making predictions.
    x_data (array-like): Data features to make predictions on.
    y_normalizer: The normalizer used for the target variable, to unscale the predictions (optional).
    unnormalize (bool): Whether to unscale the predictions using the y_normalizer. Defaults to False.

    Returns:
    tuple: Mean and standard deviation of predictions (unnormalized if specified).
    """
    # Generate predictions using the model
    predictions = model(x_data, training=False)

    # Extract mean and standard deviation from the predicted distributions
    pred_mean = predictions.mean()
    pred_stddev = predictions.stddev()

    if unnormalize and y_normalizer is not None:
        pred_mean = y_normalizer.inverse_transform(pred_mean.numpy())
        pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev.numpy()
    elif unnormalize and y_normalizer is None:
        raise ValueError(
            "y_normalizer must be provided to unnormalize the predictions."
        )
    elif not unnormalize and y_normalizer is not None:
        warnings.warn(
            "y_normalizer is provided but unnormalize is set to False. Set unnormalize to True to unnormalize the predictions."
        )

    return pred_mean, pred_stddev


def make_gp_predictions(model, x_data, y_normalizer=None, unnormalize=False):
    """
    Make predictions using the provided GP model and optionally unscale the results.

    Parameters:
    model: The trained GP model to use for making predictions.
    x_data (array-like): Data features to make predictions on.
    y_normalizer: The normalizer used for the target variable, to unscale the predictions (optional).
    unnormalize (bool): Whether to unscale the predictions using the y_normalizer. Defaults to False.

    Returns:
    array-like: Predictions (unnormalized if specified).
    """
    # Generate predictions using the model
    predictions = model(x_data, training=False)
    pred_mean = predictions.mean()
    pred_stddev = predictions.stddev()

    if unnormalize and y_normalizer is not None:
        pred_mean = y_normalizer.inverse_transform(pred_mean.numpy())
        pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev.numpy()
    elif unnormalize and y_normalizer is None:
        raise ValueError(
            "y_normalizer must be provided to unnormalize the predictions."
        )
    elif not unnormalize and y_normalizer is not None:
        warnings.warn(
            "y_normalizer is provided but unnormalize is set to False. "
            "Set unnormalize to True to unnormalize the predictions."
        )

    return pred_mean, pred_stddev


def ensemble_gp_predictions(
    model, x_data, y_normalizer=None, unnormalize=False, n_iter=100, training=False
):
    """
    Make ensemble predictions using the provided GP model and optionally unscale the results.

    Parameters:
    model: The trained GP model to use for making predictions.
    x_data (array-like): Data features to make predictions on.
    y_normalizer: The normalizer used for the target variable, to unscale the predictions (optional).
    unnormalize (bool): Whether to unscale the predictions using the y_normalizer. Defaults to False.
    n_iter (int): Number of iterations for ensemble predictions. Defaults to 100.
    training (bool): Whether to use training mode for making predictions. Defaults to False.

    Returns:
    tuple: Mean and standard deviation of predictions (unnormalized if specified).
    """
    predictions = []
    for _ in range(n_iter):
        pred = model(x_data, training=training)
        pred_mean = pred.mean().numpy()
        pred_stddev = pred.stddev().numpy()
        if unnormalize and y_normalizer is not None:
            pred_mean = y_normalizer.inverse_transform(pred_mean)
            pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev
        elif unnormalize and y_normalizer is None:
            raise ValueError(
                "y_normalizer must be provided to unnormalize the predictions."
            )
        elif not unnormalize and y_normalizer is not None:
            warnings.warn(
                "y_normalizer is provided but unnormalize is set to False. "
                "Set unnormalize to True to unnormalize the predictions."
            )
        predictions.append((pred_mean, pred_stddev))

    means, stddevs = zip(*predictions)
    mean = np.mean(means, axis=0)
    std = np.mean(stddevs, axis=0)

    return mean, std


def extract_gp_statistics_from_pdf(model, x_data, y_normalizer=None, unnormalize=False):
    """
    Extract statistics from the GP model's PDF predictions and optionally unscale the results.

    Parameters:
    model: The trained GP model to use for making predictions.
    x_data (array-like): Data features to make predictions on.
    y_normalizer: The normalizer used for the target variable, to unscale the predictions (optional).
    unnormalize (bool): Whether to unscale the predictions using the y_normalizer. Defaults to False.

    Returns:
    tuple: Mean and standard deviation of predictions (unnormalized if specified).
    """
    # Generate predictions using the model
    predictions = model(x_data, training=False)

    # Extract mean and standard deviation from the predicted distributions
    pred_mean = predictions.mean()
    pred_stddev = predictions.stddev()

    if unnormalize and y_normalizer is not None:
        pred_mean = y_normalizer.inverse_transform(pred_mean.numpy())
        pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev.numpy()
    elif unnormalize and y_normalizer is None:
        raise ValueError(
            "y_normalizer must be provided to unnormalize the predictions."
        )
    elif not unnormalize and y_normalizer is not None:
        warnings.warn(
            "y_normalizer is provided but unnormalize is set to False. "
            "Set unnormalize to True to unnormalize the predictions."
        )

    return pred_mean, pred_stddev
