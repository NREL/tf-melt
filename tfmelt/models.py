import warnings
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable


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
            "use_batch_renorm": self.use_batch_renorm,
            "output_activation": self.output_activation,
            "initializer": self.initializer,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
        }

    def initialize_layers(self):
        """Initialize the layers of the model."""
        self.create_regularizer()
        self.create_dropout_layers()
        self.create_batch_norm_layers()
        self.create_input_layer()
        self.create_output_layer()

        # Set attribute flags based on which layers are present
        self.has_batch_norm = hasattr(self, "batch_norm_layers")
        self.has_dropout = hasattr(self, "dropout_layers")
        self.has_input_dropout = hasattr(self, "input_dropout_layer")

    def create_regularizer(self):
        """Create the regularizer."""
        self.regularizer = (
            regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg)
            if (self.l1_reg > 0 or self.l2_reg > 0)
            else None
        )

    def create_dropout_layers(self):
        """Create the dropout layers."""
        if self.dropout > 0:
            self.dropout_layers = [
                Dropout(rate=self.dropout, name=f"dropout_{i}")
                for i in range(self.depth)
            ]
        if self.input_dropout > 0:
            self.input_dropout_layer = Dropout(
                rate=self.input_dropout, name="input_dropout"
            )

    def create_batch_norm_layers(self):
        """Create the batch normalization layers with optional renormalization."""
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
                for i in range(self.depth + 1)
            ]

    def create_input_layer(self):
        """Create the input layer with associated activation layer."""
        self.dense_layer_in = Dense(
            self.width,
            activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="input2bulk",
        )
        self.activation_in = Activation(self.act_fun, name="input2bulk_act")

    def create_output_layer(self):
        """Create the output layer with activation from function."""
        self.output_layer = Dense(
            self.num_outputs,
            activation=self.output_activation,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="output",
        )

    def compute_jacobian(self, x):
        """Compute the jacobian of the model outputs with respect to inputs."""
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        elif not isinstance(x, tf.Tensor):
            raise ValueError("x must be a tf.Tensor or np.ndarray")

        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self(x)
        return tape.jacobian(y_pred, x)

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
        self.dense_layers_bulk = [
            Dense(
                self.width,
                activation=None,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name=f"bulk_{i}",
            )
            for i in range(self.depth)
        ]
        self.activations_bulk = [
            Activation(self.act_fun, name=f"bulk_act_{i}") for i in range(self.depth)
        ]

    @tf.function
    def call(self, inputs):
        """Call the ANN."""
        # Apply input layer: dense -> batch norm -> activation -> input dropout
        x = self.dense_layer_in(inputs)
        x = self.batch_norm_layers[0](x) if self.has_batch_norm else x
        x = self.activation_in(x)
        x = self.input_dropout_layer(x) if self.has_input_dropout else x

        # Apply bulk layers: dense -> batch norm -> activation -> dropout
        for i in range(self.depth):
            x = self.dense_layers_bulk[i](x)
            x = self.batch_norm_layers[i + 1](x) if self.has_batch_norm else x
            x = self.activations_bulk[i](x)
            x = self.dropout_layers[i](x) if self.has_dropout else x

        # Return output layer output with activation built in
        return self.output_layer(x)


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
        if self.depth % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: depth ({self.depth}) is not divisible by layers_per_block "
                f"({self.layers_per_block}), so the last block will have "
                f"{self.depth % self.layers_per_block} layers."
            )

        self.initialize_layers()
        super(ResidualNeuralNetwork, self).build(input_shape)

    def initialize_layers(self):
        """Initialize the layers of the ResNet."""
        super(ResidualNeuralNetwork, self).initialize_layers()

        # ResNet Bulk layers
        self.dense_layers_bulk = [
            Dense(
                self.width,
                activation=None,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name=f"bulk_{i}",
            )
            for i in range(self.depth)
        ]
        self.activation_layers_bulk = [
            Activation(self.act_fun, name=f"bulk_act_{i}") for i in range(self.depth)
        ]
        # Add layers for residual connections (Add layer for every "layers per block")
        # with remainder if depth is not divisible by layers per block
        self.add_layers = [
            Add(name=f"add_{i}")
            for i in range(
                (self.depth + self.layers_per_block - 1) // self.layers_per_block
            )
        ]
        # Optional activation after the Add layers
        if self.post_add_activation:
            self.post_add_activations = [
                Activation(self.act_fun, name=f"post_add_act_{i}")
                for i in range(self.depth // 2)
            ]

    @tf.function
    def call(self, inputs):
        """Call the ResNet."""
        # Apply input layer:
        # dense -> (pre-activation) -> batch norm -> input dropout -> (post-activation)
        x = self.dense_layer_in(inputs)
        x = self.activation_in(x) if self.pre_activation else x
        x = self.batch_norm_layers[0](x) if self.has_batch_norm else x
        x = self.input_dropout_layer(x) if self.has_input_dropout else x
        x = self.activation_in(x) if not self.pre_activation else x

        # Apply bulk layers with residual connections
        for i in range(0, self.depth):
            y = x

            # Apply bulk layer:
            # dense -> (pre-activation) -> batch norm -> dropout -> (post-activation)
            x = self.dense_layers_bulk[i](x)
            x = self.activation_layers_bulk[i](x) if self.pre_activation else x
            x = self.batch_norm_layers[i + 1](x) if self.has_batch_norm else x
            x = self.dropout_layers[i](x) if self.has_dropout else x
            x = self.activation_layers_bulk[i](x) if not self.pre_activation else x

            # Add residual connection when reaching the end of a block
            if (i + 1) % self.layers_per_block == 0 or i == self.depth - 1:
                x = self.add_layers[i // self.layers_per_block]([y, x])
                x = (
                    self.post_add_activations[i // self.layers_per_block](x)
                    if self.post_add_activation
                    else x
                )

        # Return output layer output with activation built in
        return self.output_layer(x)


@register_keras_serializable(package="tfmelt")
class BayesianNeuralNetwork(MELTModel):
    def __init__(
        self,
        num_points: Optional[int] = 1,
        num_bayesian_layers: Optional[int] = None,
        do_aleatoric: Optional[bool] = False,
        aleatoric_scale_factor: Optional[float] = 5e-2,
        scale_epsilon: Optional[float] = 1e-3,
        use_batch_renorm: Optional[bool] = True,
        **kwargs,
    ):
        """
        Initialize the BayesianNeuralNetwork model.

        Args:
            num_points (int, optional): Number of Monte Carlo samples.
            num_bayesian_layers (int, optional): Number of layers to make Bayesian.
                                                Layers are counted from the output
                                                 layer backwards.
            do_aleatoric (bool, optional): Whether to include aleatoric uncertainty.
            aleatoric_scale_factor (float, optional): Scale factor for aleatoric
                                                      uncertainty.
            scale_epsilon (float, optional): Epsilon value for the scale of the
                                             aleatoric uncertainty.
            use_batch_renorm (bool, optional): Whether to use batch renormalization.
            **kwargs: Additional keyword arguments.

        """
        super(BayesianNeuralNetwork, self).__init__(**kwargs)

        self.num_points = num_points
        self.num_bayesian_layers = num_bayesian_layers
        self.do_aleatoric = do_aleatoric
        self.aleatoric_scale_factor = aleatoric_scale_factor
        self.scale_epsilon = scale_epsilon
        self.use_batch_renorm = use_batch_renorm

        # Wrap call function with tf.function if do_aleatoric if False
        if not self.do_aleatoric:
            self.call = tf.function(self.call)

        # Update config with new attributes
        self.config.update(
            {
                "num_points": self.num_points,
                "num_bayesian_layers": self.num_bayesian_layers,
                "do_aleatoric": self.do_aleatoric,
                "aleatoric_scale_factor": self.aleatoric_scale_factor,
                "scale_epsilon": self.scale_epsilon,
                "use_batch_renorm": self.use_batch_renorm,
            }
        )

    def build(self, input_shape):
        """Build the BNN."""
        if self.num_bayesian_layers is None:
            self.num_bayesian_layers = self.depth + 1
        elif self.num_bayesian_layers > self.depth + 1:
            warnings.warn(
                f"num_bayesian_layers ({self.num_bayesian_layers}) is greater than "
                f"depth + 1 ({self.depth + 1}), so setting num_bayesian_layers to "
                f"depth + 1."
            )
            self.num_bayesian_layers = self.depth + 1
        self.initialize_layers()
        super(BayesianNeuralNetwork, self).build(input_shape)

    def initialize_layers(self):
        """Initialize the layers of the BNN."""
        super(BayesianNeuralNetwork, self).initialize_layers()

        # Identify the number of bulk Bayesian layers
        num_bulk_bayesian_layers = max(
            0, self.num_bayesian_layers - (1 if self.do_aleatoric else 0)
        )
        # Identify the number of bulk Dense layers
        num_bulk_dense_layers = (
            self.depth - num_bulk_bayesian_layers - (1 if self.do_aleatoric else 0)
        )

        # Create the kernel divergence function
        self.kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(
            q, p
        ) / (self.num_points * 1.0)

        # Create a bayesian input layer if num_bayesian_layers > depth
        if self.num_bayesian_layers > self.depth:
            self.dense_layer_in = tfp.layers.DenseFlipout(
                self.width,
                kernel_divergence_fn=self.kernel_divergence_fn,
                activation=None,
                activity_regularizer=self.regularizer,
                name="input2bulk_bayesian",
            )

        # Create the Bayesian layers
        self.bayesian_layers = [
            tfp.layers.DenseFlipout(
                self.width,
                kernel_divergence_fn=self.kernel_divergence_fn,
                activation=None,
                activity_regularizer=self.regularizer,
                name=f"bayesian_{i}",
            )
            for i in range(num_bulk_bayesian_layers)
        ]
        # Create the bulk layers
        self.dense_layers_bulk = [
            Dense(
                self.width,
                activation=None,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name=f"bulk_{i}",
            )
            for i in range(num_bulk_dense_layers)
        ]
        # Create the activation layers for the layers
        self.activations_bulk = [
            Activation(self.act_fun, name=f"bulk_act_{i}") for i in range(self.depth)
        ]

        # Create the final distribution layer
        if self.do_aleatoric:
            # Create the pre-aleatoric layer based on number of Bayesian layers
            if self.num_bayesian_layers == 1:
                self.pre_aleatoric_layer_dense = Dense(
                    2 * self.num_outputs,
                    activation=None,
                    kernel_initializer=self.initializer,
                    kernel_regularizer=self.regularizer,
                    name="pre_aleatoric_dense",
                )
            else:
                self.pre_aleatoric_layer_flipout = tfp.layers.DenseFlipout(
                    2 * self.num_outputs,
                    kernel_divergence_fn=self.kernel_divergence_fn,
                    activation=None,
                    activity_regularizer=self.regularizer,
                    name="pre_aleatoric_flipout",
                )
            # Create the aleatoric layer
            self.output_layer = tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Normal(
                    loc=t[..., : self.num_outputs],
                    scale=self.scale_epsilon
                    + tf.math.softplus(
                        self.aleatoric_scale_factor * t[..., self.num_outputs :]
                    ),
                ),
                name="dist_output",
            )
        else:
            self.output_layer = tfp.layers.DenseFlipout(
                self.num_outputs,
                kernel_divergence_fn=self.kernel_divergence_fn,
                activation=self.output_activation,
                activity_regularizer=self.regularizer,
                name="output",
            )

    def call(self, inputs):
        """Call the BNN."""
        # Apply input layer: dense -> batch norm -> activation -> input dropout
        x = self.dense_layer_in(inputs)
        x = self.batch_norm_layers[0](x) if self.has_batch_norm else x
        x = self.activation_in(x)
        x = self.input_dropout_layer(x) if self.has_input_dropout else x

        # Apply bulk layers: dense -> batch norm -> activation -> dropout
        for i in range(self.depth - (1 if self.do_aleatoric else 0)):
            x = (
                self.dense_layers_bulk[i](x)
                if i < (self.depth - self.num_bayesian_layers)
                else self.bayesian_layers[i - (self.depth - self.num_bayesian_layers)](
                    x
                )
            )
            x = self.batch_norm_layers[i + 1](x) if self.has_batch_norm else x
            x = self.activations_bulk[i](x)
            x = self.dropout_layers[i](x) if self.has_dropout else x

        # Apply final distribution layer
        if self.do_aleatoric:
            if self.num_bayesian_layers > 1:
                x = self.pre_aleatoric_layer_flipout(x)
            else:
                x = self.pre_aleatoric_layer_dense(x)

        # Apply output layer
        return self.output_layer(x)

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
