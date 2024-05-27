from typing import Any, List, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import Regularizer


class DenseBlock(Model):
    """
    A DenseBlock consists of multiple dense layers with optional activation, dropout,
    and batch normalization.

    Args:
        node_list (List[int]): Number of nodes in each dense layer.
        activation (str, optional): Activation function. Defaults to "relu".
        dropout (float, optional): Dropout rate (0-1). Defaults to None.
        batch_norm (bool, optional): Apply batch normalization if True. Defaults to False.
        regularizer (Regularizer, optional): Kernel weights regularizer. Defaults to None.
        **kwargs: Extra arguments passed to the base class.

    Raises:
        AssertionError: If dropout is not within the range of [0, 1].
    """

    def __init__(
        self,
        node_list: List[int],
        activation: Optional[str] = "relu",
        dropout: Optional[float] = None,
        batch_norm: Optional[bool] = False,
        regularizer: Optional[Regularizer] = None,
        **kwargs: Any,
    ):
        super(DenseBlock, self).__init__(**kwargs)
        self.node_list = node_list
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.regularizer = regularizer

        # Validate dropout value
        if dropout is not None:
            assert 0 <= dropout < 1, "Dropout must be between 0 and 1"

        # Initialize layers: dense -> batch norm -> activation -> dropout
        self.layer_list = []
        for i, nodes in enumerate(node_list):
            layers = [
                Dense(
                    nodes,
                    activation=None,
                    kernel_regularizer=regularizer,
                    name=f"dense_{i}",
                )
            ]
            if batch_norm:
                layers.append(BatchNormalization(name=f"batch_norm_{i}"))
            if activation:
                layers.append(Activation(activation, name=f"activation_{i}"))
            if dropout is not None:
                layers.append(Dropout(dropout, name=f"dropout_{i}"))
            self.layer_list.extend(layers)

    def call(self, inputs: tf.Tensor, training: bool = False):
        """Forward pass through the dense block."""
        x = inputs
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x

    def get_config(self):
        """Returns the block configuration."""
        config = super(DenseBlock, self).get_config()
        config.update(
            {
                "node_list": self.node_list,
                "activation": self.activation,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "regularizer": self.regularizer,
            }
        )
        return config


class SingleMixtureOutput(Model):
    """
    Output layer for a single mixture density network.

    Args:
        num_outputs (int): Number of output nodes.
        output_activation (str): Activation function for the output layer.
        initializer (str): Kernel initializer.
        regularizer (Regularizer): Kernel regularizer.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self, num_outputs, output_activation, initializer, regularizer, **kwargs
    ):
        super(SingleMixtureOutput, self).__init__(**kwargs)
        self.mean_output_layer = Dense(
            num_outputs,
            activation=output_activation,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="mean_output",
        )
        self.log_var_output_layer = Dense(
            num_outputs,
            activation=None,  # No activation for log-variance
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="log_var_output",
        )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        mean_output = self.mean_output_layer(x, training=training)
        log_var_output = self.log_var_output_layer(x, training=training)
        return tf.stack([mean_output, log_var_output])


class MultipleMixturesOutput(Model):
    """
    Output layer for a mixture density network.

    Args:
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output nodes.
        initializer (str): Kernel initializer.
        regularizer (Regularizer): Kernel regularizer.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(self, num_mixtures, num_outputs, initializer, regularizer, **kwargs):

        super(MultipleMixturesOutput, self).__init__(**kwargs)
        self.mix_coeffs_layer = Dense(
            num_mixtures,
            activation="softmax",
            kernel_initializer=initializer,
            name="mix_coeffs",
        )
        self.mean_output_layer = Dense(
            num_mixtures * num_outputs,
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="mean_output",
        )
        self.log_var_output_layer = Dense(
            num_mixtures * num_outputs,
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="log_var_output",
        )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        m_coeffs = self.mix_coeffs_layer(x, training=training)
        mean_output = self.mean_output_layer(x, training=training)
        log_var_output = self.log_var_output_layer(x, training=training)
        return tf.concat([m_coeffs, mean_output, log_var_output], axis=-1)


class DefaultOutput(Model):
    """
    Default output layer with a single dense layer.

    Args:
        num_outputs (int): Number of output nodes.
        output_activation (str): Activation function for the output layer.
        initializer (str): Kernel initializer.
        regularizer (Regularizer): Kernel regularizer.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self, num_outputs, output_activation, initializer, regularizer, **kwargs
    ):
        super(DefaultOutput, self).__init__(**kwargs)
        self.output_layer = Dense(
            num_outputs,
            activation=output_activation,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="output",
        )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        return self.output_layer(x, training=training)
