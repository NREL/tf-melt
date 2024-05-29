import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss


def safe_exp(x):
    """Prevents overflow by clipping input range to reasonable values."""
    # TODO: Consider using tf.exp(x - tf.reduce_max(x)) instead
    x = tf.clip_by_value(x, clip_value_min=-20, clip_value_max=20)
    return tf.exp(x)


class SingleMixtureLoss(Loss):
    """
    Custom loss function for a single Gaussian mixture model.

    Args:
        variance_scale (float): Scaling factor for the variance loss.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(self, variance_scale=1.0, **kwargs):
        super(SingleMixtureLoss, self).__init__(**kwargs)
        self.variance_scale = variance_scale

    def call(self, y_true, y_pred):
        mean_pred = y_pred[0]
        log_var_pred = y_pred[1]

        precision = tf.exp(-log_var_pred)
        mse_loss = tf.reduce_mean(precision * tf.square(y_true - mean_pred))
        var_loss = tf.reduce_mean(log_var_pred)

        # Return the weighted sum of the MSE and variance loss
        return mse_loss + self.variance_scale * var_loss


class MultipleMixtureLoss(Loss):
    def __init__(self, num_mixtures, num_outputs, **kwargs):
        super(MultipleMixtureLoss, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures

    def call(self, y_true, y_pred):
        # TODO: develop verification check for this loss function
        # Extract the mixture coefficients, means, and log-variances
        end_mixture = self.num_mixtures
        end_mean = end_mixture + self.num_mixtures * self.num_outputs
        end_log_var = end_mean + self.num_mixtures * self.num_outputs

        m_coeffs = y_pred[:, :end_mixture]
        mean_preds = y_pred[:, end_mixture:end_mean]
        log_var_preds = y_pred[:, end_mean:end_log_var]

        # Reshape to ensure same shape as y_true replicated across mixtures
        mean_preds = tf.reshape(mean_preds, [-1, self.num_mixtures, self.num_outputs])
        log_var_preds = tf.reshape(
            log_var_preds, [-1, self.num_mixtures, self.num_outputs]
        )

        # Calculate the Gaussian probability density function for each component
        const_term = -0.5 * self.num_outputs * tf.math.log(2 * np.pi)
        inv_sigma_log = -0.5 * log_var_preds
        exp_term = (
            -0.5
            * tf.square(tf.expand_dims(y_true, 1) - mean_preds)
            / safe_exp(log_var_preds)
        )

        # form log probabilities
        log_probs = const_term + inv_sigma_log + exp_term

        # Calculate the log likelihood
        weighted_log_probs = log_probs + tf.math.log(m_coeffs[:, :, tf.newaxis])
        log_sum_exp = tf.reduce_logsumexp(weighted_log_probs, axis=1)

        log_likelihood = tf.reduce_mean(log_sum_exp)

        # Return the negative log likelihood
        return -log_likelihood
