import warnings
from typing import Any, List, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import Regularizer


class MELTBlock(Model):
    """
    Base class for a MELT block. Defines a block of dense layers with optional
    activation, dropout, and batch normalization. Forms the building block for
    various neural network blocks.

    Args:
        node_list (List[int]): Number of nodes in each dense layer.
        activation (str, optional): Activation function. Defaults to "relu".
        dropout (float, optional): Dropout rate (0-1). Defaults to None.
        batch_norm (bool, optional): Apply batch normalization if True. Defaults to False.
        use_batch_renorm (bool, optional): Use batch renormalization. Defaults to False.
        regularizer (Regularizer, optional): Kernel weights regularizer. Defaults to None.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        node_list: List[int],
        activation: Optional[str] = "relu",
        dropout: Optional[float] = None,
        batch_norm: Optional[bool] = False,
        use_batch_renorm: Optional[bool] = False,
        regularizer: Optional[Regularizer] = None,
        **kwargs: Any,
    ):
        super(MELTBlock, self).__init__(**kwargs)
        self.node_list = node_list
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_batch_renorm = use_batch_renorm
        self.regularizer = regularizer

        # Number of layers in the block
        self.num_layers = len(self.node_list)

        # Validate dropout value
        if self.dropout is not None:
            assert 0 <= self.dropout < 1, "Dropout must be between 0 and 1"

        # Initialize dense layers
        self.dense_layers = [
            Dense(
                node,
                activation=None,
                kernel_regularizer=self.regularizer,
                name=f"dense_{i}",
            )
            for i, node in enumerate(self.node_list)
        ]

        # Activation layers
        if self.activation:
            self.activation_layers = [
                Activation(self.activation, name=f"activation_{i}")
                for i in range(self.num_layers)
            ]

        # Optional dropout and batch norm layers
        if self.dropout > 0:
            self.dropout_layers = [
                Dropout(self.dropout, name=f"dropout_{i}")
                for i in range(self.num_layers)
            ]
        if self.batch_norm:
            self.batch_norm_layers = [
                BatchNormalization(
                    renorm=self.use_batch_renorm,
                    renorm_clipping=(
                        {
                            "rmax": 3,
                            "rmin": 1 / 3,
                            "dmax": 5,
                        }
                        if self.use_batch_renorm
                        else None
                    ),
                    name=f"batch_norm_{i}",
                )
                for i in range(self.num_layers)
            ]

        # Create config dictionary for serialization
        self.config = {
            "node_list": node_list,
            "activation": activation,
            "dropout": dropout,
            "batch_norm": batch_norm,
            "regularizer": regularizer,
        }

    def get_config(self):
        """Get the config dictionary"""
        config = super(MELTBlock, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config dictionary"""
        return cls(**config)


class DenseBlock(MELTBlock):
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
        **kwargs: Any,
    ):
        super(DenseBlock, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, training: bool = False):
        """Forward pass through the dense block."""
        x = inputs

        for i in range(self.num_layers):
            # dense -> batch norm -> activation -> dropout
            x = self.dense_layers[i](x, training=training)
            x = (
                self.batch_norm_layers[i](x, training=training)
                if self.batch_norm
                else x
            )
            x = self.activation_layers[i](x) if self.activation else x
            x = self.dropout_layers[i](x, training=training) if self.dropout > 0 else x

        return x


class ResidualBlock(MELTBlock):
    """
    A ResidualBlock consists of multiple dense layers with optional activation, dropout,
    and batch normalization. Residual connections are added after each block of layers.

    Args:
        layers_per_block (int, optional): Number of layers per residual block.
        pre_activation (bool, optional): Apply activation before adding residual connection.
        post_add_activation (bool, optional): Apply activation after adding residual connection.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        layers_per_block: Optional[int] = 2,
        pre_activation: Optional[bool] = False,
        post_add_activation: Optional[bool] = False,
        **kwargs: Any,
    ):
        super(ResidualBlock, self).__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

        # Warning if the number of layers is not divisible by layers_per_block
        if self.num_layers % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: Number of layers ({self.num_layers}) is not divisible by "
                f"layers_per_block ({self.layers_per_block}), so the last block will "
                f"have {self.num_layers % self.layers_per_block} layers."
            )

        # Initialize Add layers
        self.add_layers = [
            Add(name=f"add_{i}")
            # for i in range(self.num_layers // self.layers_per_block)
            for i in range(
                (self.num_layers + self.layers_per_block - 1) // self.layers_per_block
            )
        ]
        # Optional Add after activation layers
        if self.post_add_activation:
            self.post_add_activation_layers = [
                Activation(self.activation, name=f"post_add_activation_{i}")
                for i in range(self.num_layers // self.layers_per_block)
            ]

        # Update config dictionary for serialization
        self.config.update(
            {
                "layers_per_block": layers_per_block,
                "pre_activation": pre_activation,
                "post_add_activation": post_add_activation,
            }
        )

    def call(self, inputs: tf.Tensor, training: bool = False):
        """Forward pass through the residual block."""
        x = inputs

        for i in range(self.num_layers):
            y = x

            # dense -> (pre-activation) -> batch norm -> dropout -> (post-activation)
            x = self.dense_layers[i](x, training=training)
            x = self.activation_layers[i](x) if self.pre_activation else x
            x = (
                self.batch_norm_layers[i](x, training=training)
                if self.batch_norm
                else x
            )
            x = self.dropout_layers[i](x, training=training) if self.dropout > 0 else x
            x = self.activation_layers[i](x) if not self.pre_activation else x

            # Add residual connection when reaching the end of a block
            if (i + 1) % self.layers_per_block == 0 or i == self.num_layers - 1:
                x = self.add_layers[i // self.layers_per_block]([x, y])
                x = (
                    self.post_add_activation_layers[i // self.layers_per_block](x)
                    if self.post_add_activation
                    else x
                )

        return x


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
