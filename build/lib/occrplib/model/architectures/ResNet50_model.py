import tensorflow as tf
import keras
from keras.models import Sequential


# Define tensorflow model.
def create_model_ResNet50(
    verbose: bool = False,
    im_size: int = 227,
    dropout_param: float = 0.0,
    regularization_param: float = 0.0,
    cnn_layer_filters_value: int = 256,
) -> Sequential:
    """Receives:
        -
    Returns:
        Compiled ResNet50 model with input layer of 227x227x3 and output layer with 16 nodes.
    """

    # Create the model

    model = keras.models.Sequential()
    model.add(
        tf.keras.applications.ResNet50(include_top=False, input_shape=(im_size, im_size, 3), pooling="avg", weights=None),
    )
    model.add(tf.keras.layers.Dense(16, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        metrics=["accuracy"],
    )

    if verbose:
        print(model.summary())

    return model
