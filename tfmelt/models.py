import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="tfmelt")
class ArtificialNeuralNetwork(Model):
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
        softmax=None,
        sigmoid=None,
        **kwargs,
    ):
        super(ArtificialNeuralNetwork, self).__init__(**kwargs)

        self.num_feat = num_feat
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.act_fun = act_fun
        self.dropout = dropout
        self.input_dropout = dropout
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.sigmoid = sigmoid

        # Dropout layer
        self.dropout_layer = Dropout(rate=self.dropout, name="dropout")
        self.input_dropout_layer = Dropout(
            rate=self.input_dropout, name="input_dropout"
        )
        # Batch Normalization layer
        self.batch_norm_layer = BatchNormalization(name="batch_norm")
        # One Dense input layer
        self.input_layer = Dense(self.num_feat, activation=self.act_fun, name="input")
        # Connecting layer from input to bulk
        self.dense_layer_in = Dense(
            self.width, activation=self.act_fun, name="input2bulk"
        )
        # Bulk layers
        self.dense_layers_bulk = [
            Dense(self.width, activation=self.act_fun, name=f"bulk_{i}")
            for i in range(self.depth)
        ]
        # Connecting layer from bulk to output
        self.dense_layer_out = Dense(
            self.width, activation=self.act_fun, name="bulk2output"
        )
        # One Dense output layer with no activation
        self.output_layer = Dense(
            self.num_outputs,
            activation="softmax"
            if self.softmax
            else "sigmoid"
            if self.sigmoid
            else "None",
            name="output",
        )

    @tf.function
    def call(self, inputs):
        """Call the ANN."""
        x = self.input_layer(inputs)

        # Dropout after the inputs if requested
        if self.input_dropout:
            x = self.input_dropout_layer(x)
        # Dense layer connecting inputs to bulk
        x = self.dense_layer_in(x)

        # Bulk layers that are repeated
        for i in range(self.depth):
            if self.batch_norm:
                x = self.batch_norm_layer(x)
            if self.dropout:
                x = self.dropout_layer(x)
            x = self.dense_layers_bulk[i](x)

        # # Batch norm layer if requested
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        # Dropout after bulk layers if requestd
        if self.dropout:
            x = self.dropout_layer(x)
        # Dense layer connecting bulk to output
        x = self.dense_layer_out(x)

        # Batch norm layer if requested
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        # Dropout before final output if requested
        if self.dropout:
            x = self.dropout_layer(x)

        # Final output layer
        xout = self.output_layer(x)

        return xout

    def get_config(self):
        config = super(ArtificialNeuralNetwork, self).get_config()
        config.update(
            {
                "num_feat": self.num_feat,
                "width": self.width,
                "depth": self.depth,
                "act_fun": self.act_fun,
                "dropout": self.dropout,
                "input_dropout": self.input_dropout,
                "batch_norm": self.batch_norm,
                "softmax": self.softmax,
                "sigmoid": self.sigmoid,
            }
        )
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

        # One Dense input layer
        self.input_layer = Dense(self.num_feat, activation=self.act_fun, name="input")
        # Connecting layer from input to bulk
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
        x = self.input_layer(inputs)
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
