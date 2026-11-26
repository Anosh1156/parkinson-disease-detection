"""Model architectures: a small custom CNN plus VGG16 / ResNet50 transfer learners.

All models take raw ``[0, 255]`` RGB images of shape ``config.IMG_SHAPE`` and output a
single sigmoid probability ``P(parkinson)``. Preprocessing lives *inside* the model so
saved ``.keras`` files can be used directly for inference.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from . import config


@keras.saving.register_keras_serializable(package="parkinson")
class CaffePreprocess(layers.Layer):
    """RGB -> BGR and ImageNet mean subtraction (what VGG16 / ResNet50 weights expect)."""

    def call(self, x):
        mean = tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
        return tf.reverse(x, axis=[-1]) - mean


def _head(x, dense_units: int = 128):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DROPOUT)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(config.DROPOUT / 2)(x)
    return layers.Dense(1, activation="sigmoid", name="p_parkinson")(x)


# --------------------------------------------------------------------------- #
# Custom CNN (trained from scratch, deliberately small for ~190 images)
# --------------------------------------------------------------------------- #
def build_custom_cnn() -> keras.Model:
    inputs = keras.Input(shape=config.IMG_SHAPE)
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(config.L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SpatialDropout2D(0.1)(x)
    outputs = _head(x, dense_units=64)
    return keras.Model(inputs, outputs, name="custom_cnn")


# --------------------------------------------------------------------------- #
# Transfer learning
# --------------------------------------------------------------------------- #
def _build_transfer(backbone_fn: Callable, base_name: str, model_name: str,
                    weights: Optional[str] = "imagenet") -> keras.Model:
    base = backbone_fn(include_top=False, weights=weights, input_shape=config.IMG_SHAPE, name=base_name)
    base.trainable = False  # phase 1: train only the new head
    inputs = keras.Input(shape=config.IMG_SHAPE)
    x = CaffePreprocess(name="caffe_preprocess")(inputs)
    x = base(x, training=False)  # keep BatchNorm in inference mode, even when fine-tuning
    outputs = _head(x)
    return keras.Model(inputs, outputs, name=model_name)


def build_vgg16(weights: Optional[str] = "imagenet") -> keras.Model:
    return _build_transfer(keras.applications.VGG16, "vgg16_base", "vgg16", weights)


def build_resnet50(weights: Optional[str] = "imagenet") -> keras.Model:
    return _build_transfer(keras.applications.ResNet50, "resnet50_base", "resnet50", weights)


MODEL_REGISTRY: Dict[str, Callable[[], keras.Model]] = {
    "custom_cnn": build_custom_cnn,
    "vgg16": build_vgg16,
    "resnet50": build_resnet50,
}
BASE_LAYER_NAMES = {"vgg16": "vgg16_base", "resnet50": "resnet50_base"}


def get_model(name: str) -> keras.Model:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


def is_transfer_model(name: str) -> bool:
    return name in BASE_LAYER_NAMES


def unfreeze_top_layers(model: keras.Model, name: str, n_layers: Optional[int] = None) -> int:
    """Phase-2 helper: unfreeze the last ``n_layers`` of the backbone.

    BatchNorm layers stay frozen (standard practice for tiny datasets). The model
    must be re-compiled afterwards. Returns the number of trainable backbone layers.
    """
    n_layers = n_layers or config.UNFREEZE_LAYERS[name]
    base = model.get_layer(BASE_LAYER_NAMES[name])
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
    return sum(layer.trainable for layer in base.layers)
