"""Augmentation: E3 recipe (tf.data, CPU) and E7 recipe (inside the model, GPU)."""
import tensorflow as tf
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


def make_tf_augment(seed=42):
    """E3 recipe: flips, rotation, zoom-in only, brightness, contrast."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.06, fill_mode="reflect", seed=seed),
        layers.RandomZoom(height_factor=(-0.25, 0.0), seed=seed),
        layers.RandomBrightness(0.2, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.2, seed=seed),
    ], name="augment")


def make_ds(X, y, training=False, augment=None, batch=32, seed=42):
    """tf.data pipeline. Augmentation (if given) is applied to training data only."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    if training and augment is not None:
        ds = ds.map(lambda a, b: (tf.clip_by_value(
            augment(tf.cast(a, tf.float32), training=True), 0, 255), b),
            num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


class FieldAug(layers.Layer):
    """Random Gaussian blur / Gaussian noise / low-resolution degradation.
    Active only in training. Sigma, noise level and scale are drawn once per
    batch; each image is randomly assigned to blur, noise, low-res or untouched."""

    def __init__(self, img=224, p_blur=0.20, p_noise=0.20, p_lowres=0.15, **kw):
        super().__init__(**kw)
        self.img, self.p_blur, self.p_noise, self.p_lowres = img, p_blur, p_noise, p_lowres

    def get_config(self):
        cfg = super().get_config()
        cfg.update(img=self.img, p_blur=self.p_blur, p_noise=self.p_noise, p_lowres=self.p_lowres)
        return cfg

    def _gauss_blur(self, x, sigma):
        r = 5
        ax = tf.range(-r, r + 1, dtype=tf.float32)
        k1 = tf.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        k1 = k1 / tf.reduce_sum(k1)
        k2 = k1[:, None] * k1[None, :]
        k = tf.tile(k2[:, :, None, None], [1, 1, 3, 1])
        xp = tf.pad(x, [[0, 0], [r, r], [r, r], [0, 0]], mode="REFLECT")
        return tf.nn.depthwise_conv2d(xp, k, [1, 1, 1, 1], "VALID")

    def call(self, x, training=None):
        if not training:
            return x
        b = tf.shape(x)[0]
        r = tf.random.uniform((b, 1, 1, 1))
        blurred = self._gauss_blur(x, tf.random.uniform((), 0.5, 2.0))
        std = tf.random.uniform((b, 1, 1, 1), 5.0, 20.0)
        noisy = x + tf.random.normal(tf.shape(x)) * std
        small = tf.cast(self.img * tf.random.uniform((), 0.4, 0.8), tf.int32)
        lowres = tf.image.resize(tf.image.resize(x, [small, small]), [self.img, self.img])
        out = tf.where(r < self.p_blur, blurred,
              tf.where(r < self.p_blur + self.p_noise, noisy,
              tf.where(r < self.p_blur + self.p_noise + self.p_lowres, lowres, x)))
        return tf.clip_by_value(out, 0.0, 255.0)


def build_e7(base_model, img=224, seed=42):
    """Wrap an inference model with in-model augmentation (E7 recipe).
    Returns (training wrapper, inference model). Export the inference model."""
    inp = layers.Input((img, img, 3))
    x = layers.RandomFlip("horizontal_and_vertical", seed=seed)(inp)
    x = layers.RandomRotation(0.04, fill_mode="reflect", seed=seed)(x)
    x = layers.RandomZoom(height_factor=(-0.10, 0.0), seed=seed)(x)   # zoom in only
    x = layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed)(x)
    x = layers.RandomContrast(0.15, seed=seed)(x)
    x = FieldAug(img=img, name="field_aug")(x)
    out = base_model(x)
    return tf.keras.Model(inp, out, name="e7_train_wrapper"), base_model
