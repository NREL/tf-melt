import contextlib
import io
import sys

import pytest
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from tfmelt.models import ArtificialNeuralNetwork, MELTModel, ResidualNeuralNetwork

# Define expected variables
NUM_FEAT = 10
NUM_OUTPUTS = 3
WIDTH = 8
DEPTH = 2
NUM_LAYERS = (
    # 4 layers per input and hidden layer (dense, activation, dropout, batch norm)
    (DEPTH + 1) * 4
    # 1 layer for the output layer (dense with activation built-in)
    + 1
)
TOTAL_NUM_TRAINABLE_PARAMS = (
    (WIDTH * NUM_FEAT)  # input layer weights
    + WIDTH  # input layer biases
    + DEPTH * (WIDTH * WIDTH)  # hidden layers weights
    + DEPTH * WIDTH  # hidden layers biases
    + (WIDTH * NUM_OUTPUTS)  # output layer weights
    + NUM_OUTPUTS  # biases for output layer
)
TOTAL_NUM_PARAMS = TOTAL_NUM_TRAINABLE_PARAMS + 4 * WIDTH * (
    DEPTH + 1
)  # batch norm params
DATA_SIZE = 5

# create some fake data
data = tf.random.uniform((DATA_SIZE, NUM_FEAT))


@pytest.fixture
def model_config():
    return {
        "num_outputs": NUM_OUTPUTS,
        "width": WIDTH,
        "depth": DEPTH,
        "act_fun": "relu",
        "dropout": 0.5,
        "input_dropout": 0.5,
        "batch_norm": True,
        "output_activation": None,
        "initializer": "glorot_uniform",
        "l1_reg": 0.0,
        "l2_reg": 0.0,
    }


@pytest.fixture
def melt_model(model_config):
    """Create an instance of the MELTModel class."""
    return MELTModel(**model_config)


@pytest.fixture
def ann_model(model_config):
    """Create an instance of the ArtificialNeuralNetwork class."""
    return ArtificialNeuralNetwork(**model_config)


@pytest.fixture
def resnet_model(model_config):
    """Create an instance of the ResidualNeuralNetwork class."""
    resnet_config = model_config.copy()
    resnet_config["layers_per_block"] = 2
    resnet_config["pre_activation"] = True
    resnet_config["post_add_activation"] = False
    return ResidualNeuralNetwork(**model_config)


class TestMELTModel:
    def test_initialize_layers(self, melt_model):
        """Test the initialize_layers method."""
        melt_model.initialize_layers()

        # Check that the regularizer and activations have been initialized correctly
        assert hasattr(melt_model, "regularizer")
        assert hasattr(melt_model, "activation_in")

        # Check that the layers have been initialized correctly
        assert hasattr(melt_model, "dense_layer_in")
        assert hasattr(melt_model, "output_layer")

        # Check that dropout and batch norm layers have been initialized correctly
        assert hasattr(melt_model, "dropout_layers")
        assert hasattr(melt_model, "input_dropout_layer")
        assert hasattr(melt_model, "batch_norm_layers")

    def test_get_config(self, melt_model, model_config):
        """Test the get_config method."""
        config = melt_model.get_config()

        # Check that the config dictionary is correct
        assert config == model_config

    def test_from_config(self, model_config):
        """Test the from_config method."""
        model = MELTModel.from_config(model_config)

        # Check that the model has been created correctly
        assert isinstance(model, MELTModel)
        assert model.num_outputs == NUM_OUTPUTS
        assert model.width == WIDTH
        assert model.depth == DEPTH
        assert model.act_fun == "relu"
        assert model.dropout == 0.5
        assert model.input_dropout == 0.5
        assert model.batch_norm == True
        assert model.output_activation == None
        assert model.initializer == "glorot_uniform"
        assert model.l1_reg == 0.0
        assert model.l2_reg == 0.0


class TestArtificialNeuralNetwork:
    def test_build(self, ann_model):
        """Test the build method."""
        # Call the build method
        ann_model.build((None, NUM_FEAT))
        # Check that the model has been built correctly
        assert ann_model.built

    def test_output_shape(self, ann_model):
        """Test the output shape of the model."""
        # Build the model
        ann_model.build((None, NUM_FEAT))
        # Call the model and get the output
        output = ann_model.call(data)
        # Check that the output has the right shape
        assert output.shape == (DATA_SIZE, NUM_OUTPUTS)

    def test_num_parameters(self, ann_model):
        """Test the number of parameters in the model."""
        # Build the model with all the layers
        ann_model.build((None, NUM_FEAT))
        # Check that the number of parameters is as expected
        assert ann_model.count_params() == TOTAL_NUM_PARAMS

    def test_num_layers(self, ann_model):
        """Test the number of layers in the model."""
        # Build the model with all the layers
        ann_model.build((None, NUM_FEAT))
        # Check that the number of layers is as expected
        print(ann_model.layers)
        assert len(ann_model.layers) == NUM_LAYERS

    def test_initialize_layers(self, ann_model):
        """Test the initialize_layers method."""
        # Call the initialize_layers method
        ann_model.initialize_layers()

        # Check that the layers have been initialized correctly
        assert hasattr(ann_model, "dense_layer_in")
        assert hasattr(ann_model, "activation_in")
        assert hasattr(ann_model, "dense_layers_bulk")
        assert hasattr(ann_model, "activations_bulk")
        assert hasattr(ann_model, "output_layer")

        # Check that dropout and batch norm layers have been initialized correctly
        if ann_model.dropout > 0:
            assert hasattr(ann_model, "dropout_layers")
        if ann_model.input_dropout > 0:
            assert hasattr(ann_model, "input_dropout_layer")
        if ann_model.batch_norm:
            assert hasattr(ann_model, "batch_norm_layers")

    def test_get_config(self, ann_model, model_config):
        """Test the get_config method."""
        # Call the get_config method
        config = ann_model.get_config()
        # Check that the config dictionary is correct
        assert config == model_config

    def test_unused_layers(self, ann_model):
        """Test that there are no unused layers in the model."""
        # Call the build method
        ann_model.build((None, NUM_FEAT))
        # Capture the output of model.summary()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            ann_model.summary()
        summary = stream.getvalue()
        # Check that the unused layers are not in the summary
        assert (
            "unused" not in summary
        ), f"There are unused layers in the model.\n Summary: {summary}"

    def test_sigmoid_output(self, ann_model):
        """Test the output of the model with sigmoid activation."""
        ann_model.output_activation = "sigmoid"
        # Build the model with sigmoid activation
        ann_model.build((None, NUM_FEAT))
        # Call the model and get the output
        output = ann_model.call(data)
        # Check that each output is between 0 and 1
        assert tf.reduce_min(output) >= 0
        assert tf.reduce_max(output) <= 1

    def test_softmax_output(self, ann_model):
        """Test the output of the model with softmax activation."""
        ann_model.output_activation = "softmax"
        # Build the model with softmax activation
        ann_model.build((None, NUM_FEAT))
        # Call the model and get the output
        output = ann_model.call(data)
        # Check that the output sums to 1 for each sample
        assert tf.reduce_sum(output).numpy() == pytest.approx(
            DATA_SIZE * 1, sys.float_info.epsilon
        )

    def test_serialization(self, ann_model):
        """Test that the model can be serialized and deserialized."""
        # Build the model
        ann_model.build((None, NUM_FEAT))
        # Serialize the model
        serialized_model = ann_model.to_json()
        assert serialized_model

        # Deserialize the model
        deserialized_model = model_from_json(
            serialized_model,
            custom_objects={"ArtificialNeuralNetwork": ArtificialNeuralNetwork},
        )
        assert deserialized_model

        # Build the deserialized model
        deserialized_model.build((None, NUM_FEAT))

        # Check that the deserialized model is the same as the original model
        assert ann_model.get_config() == deserialized_model.get_config()
        assert ann_model.count_params() == deserialized_model.count_params()


class TestResidualNeuralNetwork:
    def test_initialize_layers(self, resnet_model):
        """Test the initialize_layers method."""
        resnet_model.initialize_layers()

        # Check that the layers have been initialized correctly
        assert hasattr(resnet_model, "dense_layers_bulk")
        assert hasattr(resnet_model, "activation_layers_bulk")
        assert hasattr(resnet_model, "add_layers")
        if resnet_model.post_add_activation:
            assert hasattr(resnet_model, "post_add_activations")

    def test_build(self, resnet_model):
        """Test the build method."""
        # Call the build method
        resnet_model.build((None, NUM_FEAT))
        # Check that the model has been built correctly
        assert resnet_model.built

    def test_call(self, resnet_model):
        """Test the call method."""
        # Build the model
        resnet_model.build((None, NUM_FEAT))
        # Call the model and get the output
        output = resnet_model.call(data)
        # Check that the output has the right shape
        assert output.shape == (DATA_SIZE, NUM_OUTPUTS)
