# mobilenet_simplified.py
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU,
    GlobalAveragePooling2D, Dense, ZeroPadding2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def _depthwise_separable_conv(inputs,
                              pointwise_conv_filters,
                              depth_multiplier=1,
                              strides=(1, 1),
                              block_id=1):
    """A single depthwise-separable conv block (MobileNetV1 style)."""
    channel_axis = -1

    if strides != (1, 1):
        x = ZeroPadding2D(((0, 1), (0, 1)), name=f"conv_pad_{block_id}")(inputs)
    else:
        x = inputs

    x = DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name=f"conv_dw_{block_id}",
    )(x)
    x = BatchNormalization(axis=channel_axis, name=f"conv_dw_{block_id}_bn")(x)
    x = ReLU(6.0, name=f"conv_dw_{block_id}_relu")(x)

    x = Conv2D(
        pointwise_conv_filters, (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name=f"conv_pw_{block_id}",
    )(x)
    x = BatchNormalization(axis=channel_axis, name=f"conv_pw_{block_id}_bn")(x)
    x = ReLU(6.0, name=f"conv_pw_{block_id}_relu")(x)
    return x


def build_mobilenet_simplified(
    input_shape=(64, 64, 1),
    num_outputs=3,
    dense_units=(512, 256, 128),
    learning_rate=0.01,
    compile_model=True,
    loss="mean_squared_error",
    metrics=("mae", "mean_absolute_error"),
):
    """
    Create and (optionally) compile a simplified MobileNetV1-like regression model.

    Args:
        input_shape: Tuple, e.g. (64, 64, 1) for grayscale.
        num_outputs: Final Dense units (for regression dims).
        dense_units: Tuple of Dense layer sizes after GAP.
        learning_rate: Optimizer LR (Adam).
        compile_model: If True, compile with given loss/metrics.
        loss: Loss name or tf.keras loss.
        metrics: Iterable of metrics.

    Returns:
        tf.keras.Model
    """
    inputs = Input(shape=input_shape)

    # Initial conv
    x = Conv2D(32, (3, 3), padding="same", use_bias=False, strides=(2, 2), name="conv1")(inputs)
    x = BatchNormalization(axis=-1, name="conv1_bn")(x)
    x = ReLU(6.0, name="conv1_relu")(x)

    # Depthwise separable conv blocks (subset of MobileNetV1)
    x = _depthwise_separable_conv(x, 64,  block_id=1)
    x = _depthwise_separable_conv(x, 128, strides=(2, 2), block_id=2)
    x = _depthwise_separable_conv(x, 128, block_id=3)
    x = _depthwise_separable_conv(x, 256, strides=(2, 2), block_id=4)
    x = _depthwise_separable_conv(x, 256, block_id=5)
    x = _depthwise_separable_conv(x, 512, strides=(2, 2), block_id=6)
    x = _depthwise_separable_conv(x, 512, block_id=7)
    x = _depthwise_separable_conv(x, 512, block_id=8)
    x = _depthwise_separable_conv(x, 512, block_id=9)
    x = _depthwise_separable_conv(x, 512, block_id=10)
    x = _depthwise_separable_conv(x, 512, block_id=11)
    x = _depthwise_separable_conv(x, 1024, strides=(2, 2), block_id=12)
    x = _depthwise_separable_conv(x, 1024, block_id=13)

    # Head
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    for i, units in enumerate(dense_units, start=1):
        x = Dense(units, activation="relu", name=f"dense_{i}")(x)

    outputs = Dense(num_outputs, activation="linear", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="mobilenet_simplified")

    if compile_model:
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))

    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_mobilenet_simplified()
    m.summary()
