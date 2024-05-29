import warnings
from itertools import groupby
from typing import List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

from tfmelt.blocks import (
    BayesianBlock,
    DefaultOutput,
    DenseBlock,
    MultipleMixturesOutput,
    ResidualBlock,
    SingleMixtureOutput,
)
from tfmelt.losses import MultipleMixtureLoss, SingleMixtureLoss


@register_keras_serializable(package="tfmelt")
class MELTModel(Model):
    def __init__(
        self,
        num_outputs: int,
        width: Optional[int] = 32,
        depth: Optional[int] = 2,
        act_fun: Optional[str] = "relu",
        dropout: Optional[float] = 0.0,
        input_dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = False,
        use_batch_renorm: Optional[bool] = False,
        output_activation: Optional[str] = None,
        initializer: Optional[str] = "glorot_uniform",
        l1_reg: Optional[float] = 0.0,
        l2_reg: Optional[float] = 0.0,
        num_mixtures: Optional[int] = 0,
        node_list: Optional[list] = None,
        **kwargs,
    ):
        """
        TF-MELT Base model.

        Args:
            num_outputs (int): Number of output units.
            width (int, optional): Width of the hidden layers.
            depth (int, optional): Number of hidden layers.
            act_fun (str, optional): Activation function for the hidden layers.
            dropout (float, optional): Dropout rate for the hidden layers.
            input_dropout (float, optional): Dropout rate for the input layer.
            batch_norm (bool, optional): Whether to use batch normalization.
            use_batch_renorm (bool, optional): Whether to use batch renormalization.
            output_activation (str, optional): Activation function for the output layer.
            initializer (str, optional): Initializer for the weights.
            l1_reg (float, optional): L1 regularization for the weights.
            l2_reg (float, optional): L2 regularization for the weights.
            do_aleatoric (bool, optional): Flag to perform aleatoric UQ.
            num_mixtures (int, optional): Number of mixtures for density networks.
            node_list (list, optional): Numbers of nodes to alternately define layers.
            **kwargs: Additional keyword arguments.

        """
        super(MELTModel, self).__init__(**kwargs)

        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.act_fun = act_fun
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.batch_norm = batch_norm
        self.use_batch_renorm = use_batch_renorm
        self.output_activation = output_activation
        self.initializer = initializer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.num_mixtures = num_mixtures
        self.node_list = node_list

        # Determine if network should be defined based on depth/width or node_list
        if self.node_list:
            self.num_layers = len(self.node_list)
            self.layer_width = self.node_list
        else:
            self.num_layers = self.depth
            self.layer_width = [self.width for i in range(self.depth)]

        # Create list for storing names of sub-layers
        self.sub_layer_names = []

        # Create config dictionary for serialization
        self.config = {
            "num_outputs": self.num_outputs,
            "width": self.width,
            "depth": self.depth,
            "act_fun": self.act_fun,
            "dropout": self.dropout,
            "input_dropout": self.input_dropout,
            "batch_norm": self.batch_norm,
            "use_batch_renorm": self.use_batch_renorm,
            "output_activation": self.output_activation,
            "initializer": self.initializer,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "num_mixtures": self.num_mixtures,
            "node_list": self.node_list,
            "num_layers": self.num_layers,
            "layer_width": self.layer_width,
        }

    def initialize_layers(self):
        """Initialize the layers of the model."""
        self.create_regularizer()
        self.create_dropout_layers()
        self.create_output_layer()

    def create_regularizer(self):
        """Create the regularizer."""
        self.regularizer = (
            regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg)
            if (self.l1_reg > 0 or self.l2_reg > 0)
            else None
        )

    def create_dropout_layers(self):
        """Create the dropout layers."""
        if self.input_dropout > 0:
            self.input_dropout_layer = Dropout(
                rate=self.input_dropout, name="input_dropout"
            )

    def create_output_layer(self):
        """Create the output layer based on the number of mixtures."""

        if self.num_mixtures == 1:
            # Single Mixture Density Network output layer
            self.output_layer = SingleMixtureOutput(
                num_outputs=self.num_outputs,
                output_activation=self.output_activation,
                initializer=self.initializer,
                regularizer=self.regularizer,
                name="single_mixture_output",
            )
            self.sub_layer_names.append("single_mixture_output")

        elif self.num_mixtures > 1:
            # Multiple Mixture Density Network output layer
            self.output_layer = MultipleMixturesOutput(
                num_mixtures=self.num_mixtures,
                num_outputs=self.num_outputs,
                initializer=self.initializer,
                regularizer=self.regularizer,
                name="multiple_mixture_output",
            )
            self.sub_layer_names.append("multiple_mixture_output")

        else:
            # Regular output layer
            self.output_layer = DefaultOutput(
                num_outputs=self.num_outputs,
                output_activation=self.output_activation,
                initializer=self.initializer,
                regularizer=self.regularizer,
                name="output",
            )
            self.sub_layer_names.append("output")

    def compute_jacobian(self, x):
        """Compute the Jacobian of the model outputs with respect to inputs."""
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        elif not isinstance(x, tf.Tensor):
            raise ValueError("x must be a tf.Tensor or np.ndarray")

        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self(x)
        return tape.jacobian(y_pred, x)

    def compile(self, optimizer="adam", loss="mse", metrics=None, **kwargs):
        """Compile the model with the appropriate loss function."""
        if self.num_mixtures == 1:
            warnings.warn(
                "Loss function is overridden when using aleatoric uncertainty. "
                "Using the aleatoric loss function."
            )
            loss = SingleMixtureLoss()
        elif self.num_mixtures > 1:
            warnings.warn(
                "Loss function is overridden when using mixture density networks. "
                "Using the mixture density loss function."
            )
            loss = MultipleMixtureLoss(self.num_mixtures, self.num_outputs)

        super(MELTModel, self).compile(optimizer, loss, metrics, **kwargs)

    def summary(self):
        """Print a summary of the model that includes sub-layers."""
        super(MELTModel, self).summary()

        # Loop over sub-layers and print summaries
        for layer_name in self.sub_layer_names:
            self.get_layer(layer_name).summary()

    def get_config(self):
        """Get the config dictionary."""
        config = super(MELTModel, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        """Create a model from a config dictionary."""
        return cls(**config)


@register_keras_serializable(package="tfmelt")
class ArtificialNeuralNetwork(MELTModel):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Artificial Neural Network model.

        Args:
            **kwargs: Additional keyword arguments.

        """
        super(ArtificialNeuralNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the ANN."""
        self.initialize_layers()
        super(ArtificialNeuralNetwork, self).build(input_shape)

    def initialize_layers(self):
        """Initialize the layers of the ANN."""
        super(ArtificialNeuralNetwork, self).initialize_layers()

        # Bulk layers
        self.dense_block = DenseBlock(
            node_list=self.layer_width,
            activation=self.act_fun,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            use_batch_renorm=self.use_batch_renorm,
            regularizer=self.regularizer,
            name="dense_block",
        )
        self.sub_layer_names.append("dense_block")

    def call(self, inputs, training=False):
        """Call the ANN."""
        # Apply input dropout
        x = (
            self.input_dropout_layer(inputs, training=training)
            if self.input_dropout > 0
            else inputs
        )

        # Apply the dense block
        x = self.dense_block(x, training=training)

        # Apply the output layer(s) and return
        return self.output_layer(x, training=training)


@register_keras_serializable(package="tfmelt")
class ResidualNeuralNetwork(MELTModel):
    def __init__(
        self,
        layers_per_block: Optional[int] = 2,
        pre_activation: Optional[bool] = True,
        post_add_activation: Optional[bool] = False,
        **kwargs,
    ):
        """
        Initialize the ResidualNeuralNetwork model.

        Args:
            layers_per_block (int, optional): Number of layers in each block.
            pre_activation (bool, optional): Whether to use pre-activation in residual
                                             blocks.
            post_add_activation (bool, optional): Whether to apply activation after
                                                  adding the residual connection.
            **kwargs: Additional keyword arguments.

        """
        super(ResidualNeuralNetwork, self).__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

        # Update config with new attributes
        self.config.update(
            {
                "layers_per_block": self.layers_per_block,
                "pre_activation": self.pre_activation,
                "post_add_activation": self.post_add_activation,
            }
        )

    def build(self, input_shape):
        """Build the ResNet."""
        if self.num_layers % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: Number of layers ({self.num_layers}) is not divisible by "
                f"layers_per_block ({self.layers_per_block}), so the last block will "
                f"have {self.num_layers % self.layers_per_block} layers."
            )

        self.initialize_layers()
        super(ResidualNeuralNetwork, self).build(input_shape)

    def initialize_layers(self):
        """Initialize the layers of the ResNet."""
        super(ResidualNeuralNetwork, self).initialize_layers()

        # Create the Residual block
        self.residual_block = ResidualBlock(
            node_list=self.layer_width,
            layers_per_block=self.layers_per_block,
            activation=self.act_fun,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            use_batch_renorm=self.use_batch_renorm,
            regularizer=self.regularizer,
            pre_activation=self.pre_activation,
            post_add_activation=self.post_add_activation,
            name="residual_block",
        )
        self.sub_layer_names.append("residual_block")

    def call(self, inputs, training=False):
        """Call the ResNet."""

        # Apply input dropout
        x = (
            self.input_dropout_layer(inputs, training=training)
            if self.input_dropout > 0
            else inputs
        )

        # Apply the Residual blocks
        x = self.residual_block(x, training=training)

        # Apply the output layer(s) and return
        return self.output_layer(x, training=training)


@register_keras_serializable(package="tfmelt")
class BayesianNeuralNetwork(MELTModel):
    def __init__(
        self,
        num_points: Optional[int] = 1,
        do_aleatoric: Optional[bool] = False,
        aleatoric_scale_factor: Optional[float] = 5e-2,
        scale_epsilon: Optional[float] = 1e-3,
        use_batch_renorm: Optional[bool] = True,
        bayesian_mask: Optional[List[bool]] = None,
        **kwargs,
    ):
        """
        Initialize the BayesianNeuralNetwork model.

        Args:
            num_points (int, optional): Number of Monte Carlo samples.
            do_aleatoric (bool, optional): Flag to perform aleatoric output.
            aleatoric_scale_factor (float, optional): Scale factor for aleatoric
                                                      uncertainty.
            scale_epsilon (float, optional): Epsilon value for the scale of the
                                             aleatoric uncertainty.
            use_batch_renorm (bool, optional): Whether to use batch renormalization.
            **kwargs: Additional keyword arguments.

        """
        super(BayesianNeuralNetwork, self).__init__(**kwargs)

        self.num_points = num_points
        self.do_aleatoric = do_aleatoric
        self.aleatoric_scale_factor = aleatoric_scale_factor
        self.scale_epsilon = scale_epsilon
        self.use_batch_renorm = use_batch_renorm
        self.bayesian_mask = bayesian_mask

        # Checks on bayesian mask and number of layers
        if len(self.bayesian_mask) > self.num_layers:
            warnings.warn(
                "Bayesian mask is longer than the number of layers, so truncating."
            )
            self.bayesian_mask = self.bayesian_mask[: self.num_layers]
        elif len(self.bayesian_mask) < self.num_layers:
            raise ValueError(
                "Bayesian mask is shorter than the number of layers."
                "Please provide a mask for each layer."
            )

        # Update config with new attributes
        self.config.update(
            {
                "num_points": self.num_points,
                "do_aleatoric": self.do_aleatoric,
                "aleatoric_scale_factor": self.aleatoric_scale_factor,
                "scale_epsilon": self.scale_epsilon,
                "use_batch_renorm": self.use_batch_renorm,
                "bayesian_mask": self.bayesian_mask,
            }
        )

    def build(self, input_shape):
        """Build the BNN."""

        self.initialize_layers()
        super(BayesianNeuralNetwork, self).build(input_shape)

    def initialize_layers(self):
        """Initialize the layers of the BNN."""
        super(BayesianNeuralNetwork, self).initialize_layers()

        # Create the Bayesian and Dense blocks based on the mask
        if self.bayesian_mask is None:
            self.num_dense_layers = 0
            self.dense_block = None
            self.bayesian_block = DenseBlock(
                num_points=self.num_points,
                node_list=self.layer_width,
                activation=self.act_fun,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                use_batch_renorm=self.use_batch_renorm,
                regularizer=self.regularizer,
                name="full_bayesian_block",
            )
            self.sub_layer_names.append("full_bayesian_block")
        else:

            self.dense_block = []
            self.bayesian_block = []

            bayes_block_idx = 0
            dense_block_idx = 0

            # Loop through the Bayesian mask and create the blocks
            idx = 0
            for is_bayesian, group in groupby(self.bayesian_mask):
                # Get the group and layer width
                group_list = list(group)
                group_len = len(group_list)
                layer_width = self.layer_width[idx : idx + group_len]
                idx += group_len

                # Create a Bayesian block or Dense block
                if is_bayesian:
                    self.bayesian_block.append(
                        BayesianBlock(
                            num_points=self.num_points,
                            node_list=layer_width,
                            activation=self.act_fun,
                            dropout=self.dropout,
                            batch_norm=self.batch_norm,
                            use_batch_renorm=self.use_batch_renorm,
                            regularizer=self.regularizer,
                            name=f"bayesian_block_{bayes_block_idx}",
                        )
                    )
                    self.sub_layer_names.append(f"bayesian_block_{bayes_block_idx}")
                    bayes_block_idx += 1
                else:
                    self.dense_block.append(
                        DenseBlock(
                            node_list=layer_width,
                            activation=self.act_fun,
                            dropout=self.dropout,
                            batch_norm=self.batch_norm,
                            use_batch_renorm=self.use_batch_renorm,
                            regularizer=self.regularizer,
                            name=f"dense_block_{dense_block_idx}",
                        )
                    )
                    self.sub_layer_names.append(f"dense_block_{dense_block_idx}")
                    dense_block_idx += 1

        # # Create the final distribution layer
        # if self.do_aleatoric:
        #     self.pre_aleatoric_layer = tfp.layers.DenseFlipout(
        #         2 * self.num_outputs,
        #         kernel_divergence_fn=self.kernel_divergence_fn,
        #         activation=None,
        #         activity_regularizer=self.regularizer,
        #         name="pre_aleatoric_flipout",
        #     )
        #     self.output_layer = tfp.layers.DistributionLambda(
        #         lambda t: tfp.distributions.Normal(
        #             loc=t[..., : self.num_outputs],
        #             scale=self.scale_epsilon
        #             + tf.math.softplus(
        #                 self.aleatoric_scale_factor * t[..., self.num_outputs :]
        #             ),
        #         ),
        #         name="dist_output",
        #     )
        # else:
        #     self.output_layer = tfp.layers.DenseFlipout(
        #         self.num_outputs,
        #         kernel_divergence_fn=self.kernel_divergence_fn,
        #         activation=self.output_activation,
        #         activity_regularizer=self.regularizer,
        #         name="output",
        #     )

    def call(self, inputs, training=False):
        """Call the BNN."""

        # Apply input dropout
        x = (
            self.input_dropout_layer(inputs, training=training)
            if self.input_dropout > 0
            else inputs
        )

        # Apply the blocks according to the Bayesian mask
        dense_idx, bayes_idx = 0, 0
        for is_bayesian, _ in groupby(self.bayesian_mask):
            if is_bayesian:
                x = self.bayesian_block[bayes_idx](x, training=training)
                bayes_idx += 1
            else:
                x = self.dense_block[dense_idx](x, training=training)
                dense_idx += 1

        # Apply the output layer(s) and return
        return self.output_layer(x, training=training)

    def negative_log_likelihood(self, y_true, y_pred):
        """Calculate the negative log likelihood."""
        return -y_pred.log_prob(y_true)

    def compile(self, optimizer="adam", loss="mse", metrics=None, **kwargs):
        """Compile the model with the appropriate loss function."""
        if self.do_aleatoric:
            warnings.warn(
                "Loss function is overridden when using aleatoric uncertainty. "
                "Using the negative log likelihood loss function."
            )
            loss = self.negative_log_likelihood

        super(BayesianNeuralNetwork, self).compile(optimizer, loss, metrics, **kwargs)
