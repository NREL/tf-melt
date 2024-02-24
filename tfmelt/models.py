from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="tfmelt")
class ArtificialNeuralNetwork(Model):
    def __init__(
        self,
        num_outputs: int,
        width: Optional[int] = 32,
        depth: Optional[int] = 2,
        act_fun: Optional[str] = "relu",
        dropout: Optional[float] = 0.0,
        input_dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = False,
        softmax: Optional[bool] = False,
        sigmoid: Optional[bool] = False,
        initializer: Optional[str] = "glorot_uniform",
        l1_reg: Optional[float] = 0.0,
        l2_reg: Optional[float] = 0.0,
        **kwargs,
    ):
        """
        Artificial Neural Network model.

        Args:
            num_outputs (int): Number of output units.
            width (int, optional): Width of the hidden layers.
            depth (int, optional): Number of hidden layers.
            act_fun (str, optional): Activation function for the hidden layers.
            dropout (float, optional): Dropout rate for the hidden layers.
            input_dropout (float, optional): Dropout rate for the input layer.
            batch_norm (bool, optional): Whether to use batch normalization.
            softmax (bool, optional): Whether to use softmax activation for the
                                      output layer.
            sigmoid (bool, optional): Whether to use sigmoid activation for the
                                      output layer.
            initializer (str, optional): Initializer for the weights.
            l1_reg (float, optional): L1 regularization for the weights.
            l2_reg (float, optional): L2 regularization for the weights.
            **kwargs: Additional keyword arguments.

        """
        super(ArtificialNeuralNetwork, self).__init__(**kwargs)

        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.act_fun = act_fun
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.initializer = initializer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        # Initialize flags for layers (to be set in build method)
        self.has_batch_norm = False
        self.has_dropout = False
        self.has_input_dropout = False

        # Create config dictionary for serialization
        self.config = {
            "num_outputs": self.num_outputs,
            "width": self.width,
            "depth": self.depth,
            "act_fun": self.act_fun,
            "dropout": self.dropout,
            "input_dropout": self.input_dropout,
            "batch_norm": self.batch_norm,
            "softmax": self.softmax,
            "sigmoid": self.sigmoid,
            "initializer": self.initializer,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
        }

    def build(self, input_shape):
        """Build the ANN."""
        if self.softmax and self.sigmoid:
            raise ValueError("Both softmax and sigmoid cannot be True.")
        self.initialize_layers()
        super(ArtificialNeuralNetwork, self).build(input_shape)

    def initialize_layers(self):
        """Initialize the layers of the ANN."""
        # Create regularizer
        regularizer = (
            regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg)
            if (self.l1_reg > 0 or self.l2_reg > 0)
            else None
        )

        # Dropout layers
        if self.dropout > 0:
            self.dropout_layers = [
                Dropout(rate=self.dropout, name=f"dropout_{i}")
                for i in range(self.depth)
            ]
        if self.input_dropout > 0:
            self.input_dropout_layer = Dropout(
                rate=self.input_dropout, name="input_dropout"
            )
        # Batch Normalization layers
        if self.batch_norm:
            self.batch_norm_layers = [
                BatchNormalization(name=f"batch_norm_{i}")
                for i in range(self.depth + 1)
            ]
        # One Dense layer connecting inputs to bulk layers
        self.dense_layer_in = Dense(
            self.width,
            activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizer,
            name="input2bulk",
        )
        self.activation_in = Activation(self.act_fun, name="input2bulk_act")
        # Bulk layers
        self.dense_layers_bulk = [
            Dense(
                self.width,
                activation=None,
                kernel_initializer=self.initializer,
                kernel_regularizer=regularizer,
                name=f"bulk_{i}",
            )
            for i in range(self.depth)
        ]
        self.activations_bulk = [
            Activation(self.act_fun, name=f"bulk_act_{i}") for i in range(self.depth)
        ]
        # One Dense output layer with activation if requested
        self.output_layer = Dense(
            self.num_outputs,
            activation=(self.get_ouput_activation()),
            kernel_initializer=self.initializer,
            kernel_regularizer=regularizer,
            name="output",
        )

        # Set attribute flags based on which layers are present
        self.has_batch_norm = hasattr(self, "batch_norm_layers")
        self.has_dropout = hasattr(self, "dropout_layers")
        self.has_input_dropout = hasattr(self, "input_dropout_layer")

    def get_ouput_activation(self):
        """Get the activation function for the output layer."""
        return "softmax" if self.softmax else "sigmoid" if self.sigmoid else None

    @tf.function
    def call(self, inputs):
        """Call the ANN."""
        # Apply input layer
        x = self.dense_layer_in(inputs)
        x = self.batch_norm_layers[0](x) if self.has_batch_norm else x
        x = self.activation_in(x)
        x = self.input_dropout_layer(x) if self.has_input_dropout else x

        # Apply bulk layers
        for i in range(self.depth):
            x = self.dense_layers_bulk[i](x)
            x = self.batch_norm_layers[i + 1](x) if self.has_batch_norm else x
            x = self.activations_bulk[i](x)
            x = self.dropout_layers[i](x) if self.has_dropout else x

        # Return output layer output with activation built in
        return self.output_layer(x)

    def get_config(self):
        """Get the config dictionary."""
        config = super(ArtificialNeuralNetwork, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="tfmelt")
class ResidualNeuralNetwork(Model):
    def __init__(
        self,
        num_feat=None,
        num_outputs=None,
        width=None,
        depth=None,
        act_fun=None,
        dropout=None,
        input_dropout=None,
        batch_norm=None,
        **kwargs,
    ):
        super(ResidualNeuralNetwork, self).__init__(**kwargs)

        self.num_feat = num_feat
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.act_fun = act_fun
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.batch_norm = batch_norm

        # Activation layer
        self.activation_layer = Activation(self.act_fun)
        # Add layer
        self.add_layer = Add()

        # One Dense layer connecting inputs to bulk layers
        self.dense_layer_in = Dense(
            self.width, activation=self.act_fun, name="input2bulk"
        )
        # Bulk layers
        self.linear_layers_bulk1 = [
            Dense(self.width, activation=None, name=f"bulk1_{i}")
            for i in range(self.depth)
        ]
        self.linear_layers_bulk2 = [
            Dense(self.width, activation=None, name=f"bulk2_{i}")
            for i in range(self.depth)
        ]
        # Connecting layer from bulk to output
        self.dense_layer_out = Dense(
            self.width, activation=self.act_fun, name="bulk2output"
        )
        # One Dense output layer with no activation
        self.output_layer = Dense(self.num_outputs, activation=None, name="output")

    @tf.function
    def call(self, inputs):
        """Call the ResNet."""
        x = inputs
        x = self.dense_layer_in(x)

        for i in range(self.depth):
            y = x

            x = self.linear_layers_bulk1[i](x)
            x = self.activation_layer(x)
            x = self.linear_layers_bulk2[i](x)

            x = self.add_layer([y, x])
            x = self.activation_layer(x)

        x = self.dense_layer_out(x)
        xout = self.output_layer(x)

        return xout

    def get_config(self):
        config = super(ResidualNeuralNetwork, self).get_config()
        config.update(
            {
                "num_feat": self.num_feat,
                "num_outputs": self.num_outputs,
                "width": self.width,
                "depth": self.depth,
                "act_fun": self.act_fun,
                "dropout": self.dropout,
                "input_dropout": self.input_dropout,
                "batch_norm": self.batch_norm,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="tfmelt")
class BayesianNeuralNetwork(Model):
    def __init__(
        self,
        num_feat=None,
        num_points=None,
        width=None,
        depth=None,
        act_fun=None,
        **kwargs,
    ):
        super(BayesianNeuralNetwork, self).__init__(**kwargs)

        self.num_feat = num_feat
        self.num_points = num_points
        self.width = width
        self.depth = depth
        self.act_fun = act_fun

        # kernel
        self.kernel = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (
            self.num_points * 1.0
        )
        # bias
        self.bias = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (
            self.num_points * 1.0
        )
        # normal distribution
        self.distribution = lambda t: tfp.distributions.Normal(
            loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])
        )

        # One Dense layer connecting inputs to bulk layers
        self.dense_layer_in = tfp.layers.DenseFlipout(
            self.width,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel,
            bias_divergence_fn=self.bias,
            activation=self.act_fun,
            name="input2bulk",
        )

        # Bulk layers
        self.dense_layers_bulk = [
            tfp.layers.DenseFlipout(
                self.width,
                bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                kernel_divergence_fn=self.kernel,
                bias_divergence_fn=self.bias,
                activation=self.act_fun,
                name=f"bulk_{i}",
            )
            for i in range(self.depth)
        ]

        # Layer creating inputs to final distribution
        self.param_layer = tfp.layers.DenseFlipout(
            2,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=self.kernel,
            bias_divergence_fn=self.bias,
            name="params",
        )

        # Final distribution layer
        self.dist_layer = tfp.layers.DistributionLambda(
            self.distribution,
            name="dist",
        )

    def call(self, inputs):
        """Call the BNN."""
        x = inputs

        # Dense layer connecting inputs to bulk
        x = self.dense_layer_in(x)

        # # Bulk layers that are repeated
        for i in range(self.depth):
            x = self.dense_layers_bulk[i](x)

        # Parameter output
        params = self.param_layer(x)

        # Final distribution layer
        dist = self.dist_layer(params)

        return dist

    def get_config(self):
        config = super(BayesianNeuralNetwork, self).get_config()
        config.update(
            {
                "num_feat": self.num_feat,
                "num_points": self.num_points,
                "width": self.width,
                "depth": self.depth,
                "act_fun": self.act_fun,
                "kernel": self.kernel,
                "bias": self.bias,
                "distribution": self.distribution,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
