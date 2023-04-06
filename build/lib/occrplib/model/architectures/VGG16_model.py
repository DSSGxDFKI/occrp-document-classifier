import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, SpatialDropout2D
from keras.models import Model, Sequential
from tensorflow import keras


# Step 1: Define tensorflow model.
def create_model_VGG16(
    verbose: bool = False,
    im_size: int = 224,
    dropout_param: float = 0.0,
    regularization_param: float = 0.0,
    cnn_layer_filters_value: int = 256,
) -> Sequential:
    """
    Creates the VGG16 model.

    Args:
        verbose (bool, optional): Verbose feedback, defaults to False.

    Returns
        Compiled VGG16 model with input layer of 224x224x3 and output layer with 16 nodes.
    """

    # TODO Add dropout layers.
    # TODO Add regularization parameter.
    # TODO Add variability to last-cnn layer filter No.

    # transform images
    # img_dim = (im_size, im_size, 3)
    # img_input = Input(shape=img_dim)

    # Block 1
    # relu replace all negative values with 0
    # padding="same" means that output image has smimilar size to input image
    # 64 is number of filter, 3x3 is size of filter
    # reduce image size, stride means how much we want to jump horizontally and vertically
    model = keras.models.Sequential(
        [
            # Block 1
            Conv2D(64, (3, 3), input_shape=(im_size, im_size, 3), activation="relu", padding="same", name="block1_conv1"),
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                name="block1_conv2",
                kernel_regularizer=keras.regularizers.l2(l=regularization_param),
            ),
            MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"),
            SpatialDropout2D(rate=dropout_param),
            # Block 2
            Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1"),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                name="block2_conv2",
                kernel_regularizer=keras.regularizers.l2(l=regularization_param),
            ),
            MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"),
            SpatialDropout2D(rate=dropout_param),
            # Block 3
            Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1"),
            Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2"),
            Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3"),
            MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"),
            # Block 4
            Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1"),
            Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2"),
            Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3"),
            MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"),
            # Block 5
            Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1"),
            Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2"),
            Conv2D(cnn_layer_filters_value, (3, 3), activation="relu", padding="same", name="conv2d_last"),
            MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool"),
            # Classification block
            Flatten(name="flatten"),
            Dense(4096, activation="relu", name="fc1"),  # fully connected layers, 4096 number of neurons
            Dense(4096, activation="relu", name="fc2"),
            Dense(16, activation="softmax", name="prediction"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if verbose:
        print(model.summary())

    return model
