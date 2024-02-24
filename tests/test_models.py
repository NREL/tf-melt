# test_models.py
import contextlib
import io
import sys

import pytest
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from tfmelt.models import ArtificialNeuralNetwork

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
def ann_model():
    return ArtificialNeuralNetwork(
        num_outputs=NUM_OUTPUTS,
        width=WIDTH,
        depth=DEPTH,
        act_fun="relu",
        dropout=0.5,
        input_dropout=0.5,
        batch_norm=True,
        softmax=False,
        sigmoid=False,
        initializer="glorot_uniform",
    )


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

    def test_get_config(self, ann_model):
        """Test the get_config method."""
        # Call the get_config method
        config = ann_model.get_config()
        # Check that the config dictionary is correct
        assert config == ann_model.config

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
        ann_model.sigmoid = True
        # Build the model with sigmoid activation
        ann_model.build((None, NUM_FEAT))
        # Call the model and get the output
        output = ann_model.call(data)
        # Check that each output is between 0 and 1
        assert tf.reduce_min(output) >= 0
        assert tf.reduce_max(output) <= 1

    def test_softmax_output(self, ann_model):
        """Test the output of the model with softmax activation."""
        ann_model.softmax = True
        # Build the model with softmax activation
        ann_model.build((None, NUM_FEAT))
        # Call the model and get the output
        output = ann_model.call(data)
        # Check that the output sums to 1 for each sample
        assert tf.reduce_sum(output).numpy() == pytest.approx(
            DATA_SIZE * 1, sys.float_info.epsilon
        )

    def test_softmax_sigmoid_output(self, ann_model):
        """Test that a ValueError is raised when both softmax and sigmoid are True."""
        ann_model.softmax = True
        ann_model.sigmoid = True
        # Check that a ValueError is raised when both softmax and sigmoid are True
        with pytest.raises(ValueError):
            ann_model.build((None, NUM_FEAT))

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
