import warnings
from typing import Any, List, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import Regularizer


def get_kernel_divergence_fn(num_points):
    """
    Get the kernel divergence function.

    Args:
        num_points (int): Number of points in kernel divergence.

    Returns:
        Callable: Kernel divergence function.
    """

    def kernel_divergence_fn(q, p, _):
        return tfp.distributions.kl_divergence(q, p) / tf.cast(num_points, tf.float32)

    return kernel_divergence_fn


class MELTBlock(Model):
    """
    Base class for a MELT block. Defines a block of dense layers with optional
    activation, dropout, and batch normalization. Forms the building block for
    various neural network blocks.

    Args:
        node_list (List[int]): Number of nodes in each dense layer. The length of
                               the list determines the number of layers.
        activation (str, optional): Activation function. If None,
                                    no activation is applied (linear). Defaults to
                                    "relu".
        dropout (float, optional): Dropout rate (0-1). Defaults to None.
        batch_norm (bool, optional): Apply batch normalization if True. Defaults
                                     to False.
        use_batch_renorm (bool, optional): Use batch renormalization. Defaults to False.
        regularizer (Regularizer, optional): Kernel weights regularizer. Defaults
                                             to None.
        initializer (str, optional): String defining the kernel initializer. Defaults
                                     to "glorot_uniform".
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
        initializer: Optional[str] = "glorot_uniform",
        **kwargs: Any,
    ):
        super(MELTBlock, self).__init__(**kwargs)

        self.node_list = node_list
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_batch_renorm = use_batch_renorm
        self.regularizer = regularizer
        self.initializer = initializer

        # Number of layers in the block
        self.num_layers = len(self.node_list)

        # Validate dropout value
        if self.dropout is not None:
            assert 0 <= self.dropout < 1, "Dropout must be between 0 and 1"

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
            "node_list": self.node_list,
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "regularizer": self.regularizer,
            "initializer": self.initializer,
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
        **kwargs: Extra arguments passed to the base class.

    Raises:
        AssertionError: If dropout is not within the range of [0, 1].
    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        super(DenseBlock, self).__init__(**kwargs)

        # Initialize dense layers
        self.dense_layers = [
            Dense(
                node,
                activation=None,
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer,
                name=f"dense_{i}",
            )
            for i, node in enumerate(self.node_list)
        ]

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
        layers_per_block (int, optional): Number of layers per residual block. Defaults
                                          to 2.
        pre_activation (bool, optional): Apply activation before adding residual
                                         connection. Defaults to False.
        post_add_activation (bool, optional): Apply activation after adding residual
                                              connection. Defaults to False.
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

        # Initialize dense layers
        self.dense_layers = [
            Dense(
                node,
                activation=None,
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer,
                name=f"dense_{i}",
            )
            for i, node in enumerate(self.node_list)
        ]

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


class BayesianBlock(MELTBlock):
    """
    A BayesianBlock consists of multiple Bayesian dense layers with optional activation,
    dropout, and batch normalization. The layers are implemented using the Flipout
    variational layer.

    Args:
        num_points (int, optional): Number of Monte Carlo samples. Defaults to 1.
        use_batch_renorm (bool, optional): Use batch renormalization. Defaults to True.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        num_points: Optional[int] = 1,
        use_batch_renorm: Optional[bool] = True,
        **kwargs: Any,
    ):
        super(BayesianBlock, self).__init__(**kwargs)

        self.num_points = num_points
        self.use_batch_renorm = use_batch_renorm

        # Create kernel divergence function
        kernel_divergence_fn = get_kernel_divergence_fn(self.num_points)

        # Initialize Bayesian layers
        self.bayesian_layers = [
            tfp.layers.DenseFlipout(
                node,
                activation=None,
                kernel_divergence_fn=kernel_divergence_fn,
                activity_regularizer=self.regularizer,
                name=f"bayesian_{i}",
            )
            for i, node in enumerate(self.node_list)
        ]

        # Update config dictionary for serialization
        self.config.update(
            {
                "num_points": num_points,
                "use_batch_renorm": use_batch_renorm,
            }
        )

    def call(self, inputs: tf.Tensor, training: bool = False):
        """Forward pass through the Bayesian block."""

        x = inputs

        for i in range(self.num_layers):
            # bayesian -> batch norm -> activation -> dropout
            x = self.bayesian_layers[i](x, training=training)
            x = (
                self.batch_norm_layers[i](x, training=training)
                if self.batch_norm
                else x
            )
            x = self.activation_layers[i](x) if self.activation else x
            x = self.dropout_layers[i](x, training=training) if self.dropout > 0 else x

        return x


class SingleMixtureOutput(Model):
    """
    Output layer for a single mixture density network.

    Args:
        num_outputs (int): Number of output nodes. The output layer will have twice the
                           number of nodes for the mean and log-variance.
        output_activation (str, optional): Activation function for the output layer.
                                           Defaults to None.
        initializer (str, optional): Kernel initializer. Defaults to "glorot_uniform".
        regularizer (Regularizer, optional): Kernel regularizer. Defaults to None.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        num_outputs: int,
        output_activation: Optional[str] = None,
        initializer: Optional[str] = "glorot_uniform",
        regularizer: Optional[Regularizer] = None,
        **kwargs,
    ):
        super(SingleMixtureOutput, self).__init__(**kwargs)

        self.num_outputs = num_outputs
        self.output_activation = output_activation
        self.initializer = initializer
        self.regularizer = regularizer

        # Update config dictionary for serialization
        self.config = {
            "num_outputs": self.num_outputs,
            "output_activation": self.output_activation,
            "initializer": self.initializer,
            "regularizer": self.regularizer,
        }

        self.mean_output_layer = Dense(
            self.num_outputs,
            activation=self.output_activation,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="mean_output",
        )
        self.log_var_output_layer = Dense(
            self.num_outputs,
            activation=None,  # No activation for log-variance
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="log_var_output",
        )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        mean_output = self.mean_output_layer(x, training=training)
        log_var_output = self.log_var_output_layer(x, training=training)
        return tf.stack([mean_output, log_var_output])

    def get_config(self):
        """Get the config dictionary"""
        config = super(SingleMixtureOutput, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config dictionary"""
        return cls(**config)


class MultipleMixturesOutput(Model):
    """
    Output layer for a mixture density network.

    Args:
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output nodes.
        initializer (str, optional): Kernel initializer. Defaults to "glorot_uniform".
        regularizer (Regularizer, optional): Kernel regularizer. Defaults to None.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        num_mixtures: int,
        num_outputs: int,
        initializer: Optional[str] = "glorot_uniform",
        regularizer: Optional[Regularizer] = None,
        **kwargs,
    ):
        super(MultipleMixturesOutput, self).__init__(**kwargs)

        self.num_mixtures = num_mixtures
        self.num_outputs = num_outputs
        self.initializer = initializer
        self.regularizer = regularizer

        # Update config dictionary for serialization
        self.config = {
            "num_mixtures": self.num_mixtures,
            "num_outputs": self.num_outputs,
            "initializer": self.initializer,
            "regularizer": self.regularizer,
        }

        self.mix_coeffs_layer = Dense(
            self.num_mixtures,
            activation="softmax",
            kernel_initializer=self.initializer,
            name="mix_coeffs",
        )
        self.mean_output_layer = Dense(
            self.num_mixtures * self.num_outputs,
            activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="mean_output",
        )
        self.log_var_output_layer = Dense(
            self.num_mixtures * self.num_outputs,
            activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="log_var_output",
        )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        m_coeffs = self.mix_coeffs_layer(x, training=training)
        mean_output = self.mean_output_layer(x, training=training)
        log_var_output = self.log_var_output_layer(x, training=training)
        return tf.concat([m_coeffs, mean_output, log_var_output], axis=-1)

    def get_config(self):
        """Get the config dictionary"""
        config = super(MultipleMixturesOutput, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config dictionary"""
        return cls(**config)


class DefaultOutput(Model):
    """
    Default output layer with a single dense layer.

    Args:
        num_outputs (int): Number of output nodes.
        output_activation (str, optional): Activation function for the output layer.
                                           Defaults to None.
        initializer (str, optional): Kernel initializer. Defaults to "glorot_uniform".
        regularizer (Regularizer, optional): Kernel regularizer. Defaults to None.
        bayesian (bool, optional): Use Bayesian layer if True. Defaults to False.
        num_points (int, optional): Number of samples. Defaults to 1.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        num_outputs,
        output_activation: Optional[str] = None,
        initializer: Optional[str] = "glorot_uniform",
        regularizer: Optional[Regularizer] = None,
        bayesian: Optional[bool] = False,
        num_points: Optional[int] = 1,
        **kwargs,
    ):
        super(DefaultOutput, self).__init__(**kwargs)

        self.num_outputs = num_outputs
        self.output_activation = output_activation
        self.initializer = initializer
        self.regularizer = regularizer
        self.bayesian = bayesian
        self.num_points = num_points

        # Update config dictionary for serialization
        self.config = {
            "num_outputs": self.num_outputs,
            "output_activation": self.output_activation,
            "initializer": self.initializer,
            "regularizer": self.regularizer,
            "bayesian": self.bayesian,
            "num_points": self.num_points,
        }

        # Create kernel divergence function
        if bayesian:
            kernel_divergence_fn = get_kernel_divergence_fn(self.num_points)

        if bayesian:
            self.output_layer = tfp.layers.DenseFlipout(
                self.num_outputs,
                kernel_divergence_fn=kernel_divergence_fn,
                activation=self.output_activation,
                activity_regularizer=self.regularizer,
                name="bayesian_output",
            )
        else:
            self.output_layer = Dense(
                self.num_outputs,
                activation=self.output_activation,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name="output",
            )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        return self.output_layer(x, training=training)

    def get_config(self):
        """Get the config dictionary"""
        config = super(DefaultOutput, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config dictionary"""
        return cls(**config)


class BayesianAleatoricOutput(Model):
    """
    Output layer for a Bayesian neural network with aleatoric uncertainty.

    Args:
        num_outputs (int): Number of output nodes.
        num_points (int): Number of Monte Carlo samples.
        regularizer (Regularizer, optional): Kernel regularizer. Defaults to None.
        scale_epsilon (float, optional): Epsilon value for scale parameter. Defaults to
                                         1e-3.
        aleatoric_scale_factor (float, optional): Scaling factor for aleatoric
                                                  uncertainty. Defaults to 5e-2.
        **kwargs: Extra arguments passed to the base class.
    """

    def __init__(
        self,
        num_outputs: int,
        num_points: int,
        regularizer: Optional[Regularizer] = None,
        scale_epsilon: Optional[float] = 1e-3,
        aleatoric_scale_factor: Optional[float] = 5e-2,
        **kwargs,
    ):
        super(BayesianAleatoricOutput, self).__init__(**kwargs)

        self.num_outputs = num_outputs
        self.num_points = num_points
        self.regularizer = regularizer
        self.scale_epsilon = scale_epsilon
        self.aleatoric_scale_factor = aleatoric_scale_factor

        # Update config dictionary for serialization
        self.config = {
            "num_outputs": self.num_outputs,
            "num_points": self.num_points,
            "regularizer": self.regularizer,
            "scale_epsilon": self.scale_epsilon,
            "aleatoric_scale_factor": self.aleatoric_scale_factor,
        }

        # Create kernel divergence function
        kernel_divergence_fn = get_kernel_divergence_fn(self.num_points)

        self.pre_aleatoric_layer = tfp.layers.DenseFlipout(
            2 * self.num_outputs,
            kernel_divergence_fn=kernel_divergence_fn,
            activation=None,
            activity_regularizer=self.regularizer,
            name="pre_aleatoric",
        )

        self.output_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[..., : self.num_outputs],
                scale=self.scale_epsilon
                + tf.math.softplus(
                    self.aleatoric_scale_factor * t[..., self.num_outputs :]
                ),
            ),
            name="distribution_output",
        )

    def call(self, x, training=False):
        """Forward pass through the output layer."""
        x = self.pre_aleatoric_layer(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        """Get the config dictionary"""
        config = super(BayesianAleatoricOutput, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config dictionary"""
        return cls(**config)
