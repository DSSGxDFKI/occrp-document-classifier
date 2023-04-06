import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

# Define tensorflow model.
def create_model_AlexNet(
    verbose: bool = False,
    im_size: int = 227,
    dropout_param: float = 0.0,
    regularization_param: float = 0.0,
    cnn_layer_filters_value: int = 256,
) -> Sequential:
    """Receives:
        -
    Returns:
        Compiled AlexNet model with input layer of 227x227x3 and output layer with 16 nodes.
    """

    # Create the model
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=(4, 4),
                activation="relu",
                input_shape=(im_size, im_size, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=regularization_param),
            ),  # NOQA E501
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=dropout_param),
            keras.layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=regularization_param),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=dropout_param),
            keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=regularization_param),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=regularization_param),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                filters=cnn_layer_filters_value,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=regularization_param),
                name="conv2d_last",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(16, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        metrics=["accuracy"],
    )

    if verbose:
        print(model.summary())

    return model
