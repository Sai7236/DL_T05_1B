"""Model definitions: scratch CNN baseline and MobileNetV2 transfer model."""
import tensorflow as tf
from tensorflow.keras import layers


def build_scratch_cnn(img=224, num_classes=10):
    """Small CNN trained from scratch (about 100k parameters)."""
    inp = layers.Input((img, img, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (16, 32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="scratch_cnn")


def build_mobilenet(img=224, num_classes=10, alpha=0.35, dropout=0.3, weights="imagenet"):
    """MobileNetV2 backbone (width multiplier alpha) + new softmax head.
    Rescaling is inside the model, so the TFLite file takes raw 0-255 pixels."""
    inp = layers.Input((img, img, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inp)           # MobileNet expects [-1, 1]
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img, img, 3), alpha=alpha, include_top=False,
        weights=weights, name="backbone")
    base.trainable = False                                    # head-only first
    x = base(x, training=False)                               # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name=f"mobilenetv2_{alpha}")
