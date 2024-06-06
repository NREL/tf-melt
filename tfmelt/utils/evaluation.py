import warnings
from typing import Any, Optional

import numpy as np
import tensorflow as tf


def make_predictions(
    model,
    x_data,
    y_normalizer: Optional[Any] = None,
    unnormalize: Optional[bool] = False,
    training: Optional[bool] = False,
):
    """
    Make predictions using the provided model and optionally unscale the results.

    Args:
        model: The trained model to use for making predictions.
        x_data (array-like): Data features to make predictions on.
        y_normalizer (scaler, optional): The normalizer used for the target variable, to
                                 unscale the predictions. Defaults to None.
        unnormalize (bool, optional): Whether to unscale the predictions using the
                                      y_normalizer. Defaults to False.
        training (bool, optional): Whether to use training mode for making predictions.
                        Defaults to False.

    Returns:
        array-like: Predictions (unnormalized if specified).
    """
    # TODO: fix the y_normalizer typing to be more specific

    # Make predictions (special care for mixture density networks)
    if model.num_mixtures > 1:
        pred_array = model(x_data, training=training)
        m_coeffs = pred_array[:, : model.num_mixtures]
        mean_preds = pred_array[
            :,
            model.num_mixtures : model.num_mixtures
            + model.num_mixtures * model.num_outputs,
        ]
        log_var_preds = pred_array[
            :, model.num_mixtures + model.num_mixtures * model.num_outputs :
        ]

        # Reshape mean and log variances to separate the mixture components
        mean_preds = tf.reshape(mean_preds, [-1, model.num_mixtures, model.num_outputs])
        log_var_preds = tf.reshape(
            log_var_preds, [-1, model.num_mixtures, model.num_outputs]
        )

        # Compute the weighted mean and total variance of the predictions
        m_coeffs = m_coeffs[:, :, np.newaxis]
        mean_preds_weighted = np.sum(mean_preds * m_coeffs, axis=1)
        variance_preds_weighted = (
            np.sum((mean_preds**2 + np.exp(log_var_preds)) * m_coeffs, axis=1)
            - mean_preds_weighted**2
        )
        std_preds_weighted = np.sqrt(variance_preds_weighted)

        predictions = mean_preds_weighted
        std_pred = std_preds_weighted

    elif model.num_mixtures == 1:
        predictions, log_var_pred = model(x_data, training=training)
        std_pred = np.sqrt(np.exp(log_var_pred))
    else:
        predictions = model(x_data, training=training)
        std_pred = None

    # Unscale the results if required
    if unnormalize and y_normalizer is not None:
        predictions = y_normalizer.inverse_transform(predictions)
        if std_pred is not None:
            std_pred = np.float32(y_normalizer.scale_) * std_pred
    elif unnormalize and y_normalizer is None:
        raise ValueError(
            "y_normalizer must be provided to unnormalize the predictions."
        )
    elif not unnormalize and y_normalizer is not None:
        warnings.warn(
            "y_normalizer is provided but unnormalize is set to False. "
            "Set unnormalize to True to unnormalize the predictions."
        )

    if model.num_mixtures > 1:
        return predictions, std_pred
    elif model.num_mixtures == 1:
        return predictions, std_pred
    else:
        return predictions


def ensemble_predictions(
    model,
    x_data,
    y_normalizer: Optional[Any] = None,
    unnormalize: Optional[bool] = False,
    n_iter: Optional[int] = 100,
    training: Optional[bool] = False,
):
    """
    Make ensemble predictions using the provided model and optionally unscale the
    results. This is useful for models with probabilistic evaluation such as Bayesian
    Neural Networks or Artificial Neural Networks using Monte Carlo Dropout.

    Args:
        model: The trained model to use for making predictions.
        x_data (array-like): Data features to make predictions on.
        y_normalizer (optional): The normalizer used for the target variable, to
                                 unscale the predictions. Defaults to None.
        unnormalize (bool): Whether to unscale the predictions using the y_normalizer.
                            Defaults to False.
        n_iter (int): Number of iterations for ensemble predictions. Defaults to 100.
        training (bool): Whether to use training mode for making predictions.
                         Defaults to False.

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


def extract_statistics_from_pdf(
    model,
    x_data,
    y_normalizer: Optional[Any] = None,
    unnormalize: Optional[bool] = False,
):
    """
    Extract statistics from the model's PDF predictions and optionally unscale the
    results. This is useful for models that output a distribution as the prediction,
    such as a Bayesian Neural Network with aleatoric uncertainty enabled.

    Args:
        model: The trained model to use for making predictions.
        x_data (array-like): Data features to make predictions on.
        y_normalizer (optional): The normalizer used for the target variable, to unscale
                                the predictions. Defaults to None.
        unnormalize (bool): Whether to unscale the predictions using the y_normalizer.
                            Defaults to False.

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
            "y_normalizer is provided but unnormalize is set to False. Set unnormalize "
            "to True to unnormalize the predictions."
        )

    return pred_mean, pred_stddev


# def make_gp_predictions(model, x_data, y_normalizer=None, unnormalize=False):
#     # TODO: Add GP models to the toolbox. Currently unused.
#     """
#     Make predictions using the provided GP model and optionally unscale the results.

#     Args:
#         model: The trained GP model to use for making predictions.
#         x_data (array-like): Data features to make predictions on.
#         y_normalizer (optional): The normalizer used for the target variable, to
#                                 unscale the predictions.
#         unnormalize (bool): Whether to unscale the predictions using the y_normalizer.
#                             Defaults to False.

#     Returns:
#         array-like: Predictions (unnormalized if specified).
#     """
#     # Generate predictions using the model
#     predictions = model(x_data, training=False)
#     pred_mean = predictions.mean()
#     pred_stddev = predictions.stddev()

#     if unnormalize and y_normalizer is not None:
#         pred_mean = y_normalizer.inverse_transform(pred_mean.numpy())
#         pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev.numpy()
#     elif unnormalize and y_normalizer is None:
#         raise ValueError(
#             "y_normalizer must be provided to unnormalize the predictions."
#         )
#     elif not unnormalize and y_normalizer is not None:
#         warnings.warn(
#             "y_normalizer is provided but unnormalize is set to False. "
#             "Set unnormalize to True to unnormalize the predictions."
#         )

#     return pred_mean, pred_stddev


# def ensemble_gp_predictions(
#     model,
#     x_data,
#     y_normalizer: Optional[Any] = None,
#     unnormalize: Optional[bool] = False,
#     n_iter: Optional[int] = 100,
#     training: Optional[bool] = False,
# ):
#     # TODO: Add GP models to the toolbox. Currently unused.
#     """
#     Make ensemble predictions using the provided GP model and optionally unscale the
#     results.

#     Args:
#         model: The trained GP model to use for making predictions.
#         x_data (array-like): Data features to make predictions on.
#         y_normalizer (optional): The normalizer used for the target variable, to
#                                 unscale the predictions. Defaults to None.
#         unnormalize (bool): Whether to unscale the predictions using the y_normalizer.
#                             Defaults to False.
#         n_iter (int): Number of iterations for ensemble predictions. Defaults to 100.
#         training (bool): Whether to use training mode for making predictions.
#                         Defaults to False.

#     Returns:
#         tuple: Mean and standard deviation of predictions (unnormalized if specified).
#     """
#     predictions = []
#     for _ in range(n_iter):
#         pred = model(x_data, training=training)
#         pred_mean = pred.mean().numpy()
#         pred_stddev = pred.stddev().numpy()
#         if unnormalize and y_normalizer is not None:
#             pred_mean = y_normalizer.inverse_transform(pred_mean)
#             pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev
#         elif unnormalize and y_normalizer is None:
#             raise ValueError(
#                 "y_normalizer must be provided to unnormalize the predictions."
#             )
#         elif not unnormalize and y_normalizer is not None:
#             warnings.warn(
#                 "y_normalizer is provided but unnormalize is set to False. "
#                 "Set unnormalize to True to unnormalize the predictions."
#             )
#         predictions.append((pred_mean, pred_stddev))

#     means, stddevs = zip(*predictions)
#     mean = np.mean(means, axis=0)
#     std = np.mean(stddevs, axis=0)

#     return mean, std


# def extract_gp_statistics_from_pdf(
#     model,
#     x_data,
#     y_normalizer=None,
#     unnormalize=False,
# ):
#     # TODO: Add GP models to the toolbox. Currently unused.
#     """
#     Extract statistics from the GP model's PDF predictions and optionally unscale the
#     results.

#     Args:
#         model: The trained GP model to use for making predictions.
#         x_data (array-like): Data features to make predictions on.
#         y_normalizer (optional): The normalizer used for the target variable, to
#                                 unscale the predictions.
#         unnormalize (bool): Whether to unscale the predictions using the y_normalizer.
#                             Defaults to False.

#     Returns:
#         tuple: Mean and standard deviation of predictions (unnormalized if specified).
#     """
#     # Generate predictions using the model
#     predictions = model(x_data, training=False)

#     # Extract mean and standard deviation from the predicted distributions
#     pred_mean = predictions.mean()
#     pred_stddev = predictions.stddev()

#     if unnormalize and y_normalizer is not None:
#         pred_mean = y_normalizer.inverse_transform(pred_mean.numpy())
#         pred_stddev = np.float32(y_normalizer.scale_) * pred_stddev.numpy()
#     elif unnormalize and y_normalizer is None:
#         raise ValueError(
#             "y_normalizer must be provided to unnormalize the predictions."
#         )
#     elif not unnormalize and y_normalizer is not None:
#         warnings.warn(
#             "y_normalizer is provided but unnormalize is set to False. "
#             "Set unnormalize to True to unnormalize the predictions."
#         )

#     return pred_mean, pred_stddev
