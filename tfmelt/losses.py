import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss


def safe_exp(x):
    """
    Prevents overflow by clipping input range to reasonable values.

    The function clips the input range as:

    .. math::
        x = \text{tf.clip_by_value}(x, \text{clip_value_min}=-20, \text{clip_value_max}
        =20)

    Args:
        x (tensor): Input tensor.

    """
    # TODO: Consider using tf.exp(x - tf.reduce_max(x)) instead
    x = tf.clip_by_value(x, clip_value_min=-20, clip_value_max=20)
    return tf.exp(x)


class MixtureDensityLoss(Loss):
    """
    Loss function for the Mixture Density Network (MDN) model. Computes the negative log
    likelihood using the weighted average of the Gaussian mixture model components.

    Args:
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output dimensions.
    """

    def __init__(self, num_mixtures, num_outputs, **kwargs):
        super(MixtureDensityLoss, self).__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.num_outputs = num_outputs

    def call(self, y_true, y_pred):
        # TODO: Determine if the constant terms provide any benefit
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
        const_term = -0.5 * self.num_outputs * tf.math.log(2 * tf.constant(np.pi))
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
