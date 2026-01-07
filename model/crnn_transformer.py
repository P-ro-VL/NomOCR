from config import HEIGHT, WIDTH
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Reshape, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.models import Model

class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for (B, T, D) sequences."""
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, x):
        # x: (B, T, D)
        t = tf.shape(x)[1]
        d = self.d_model

        positions = tf.cast(tf.range(t)[:, tf.newaxis], tf.float32)  # (T, 1)
        div_term = tf.exp(
            tf.cast(tf.range(0, d, 2), tf.float32) * (-tf.math.log(10000.0) / tf.cast(d, tf.float32))
        )  # (D/2,)

        pe_sin = tf.sin(positions * div_term)  # (T, D/2)
        pe_cos = tf.cos(positions * div_term)  # (T, D/2)
        pe = tf.concat([pe_sin, pe_cos], axis=-1)  # (T, D)
        pe = pe[tf.newaxis, :, :]  # (1, T, D)

        return x + pe


def transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float, name: str):
    # Self-attention
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
        name=f"{name}_mha"
    )(x, x)
    x = Add(name=f"{name}_attn_add")([x, attn_out])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_attn_ln")(x)

    # Feed-forward
    ff = Dense(ff_dim, activation="gelu", name=f"{name}_ff1")(x)
    ff = Dropout(dropout, name=f"{name}_ff_dropout")(ff)
    ff = Dense(d_model, name=f"{name}_ff2")(ff)

    x = Add(name=f"{name}_ff_add")([x, ff])
    x = LayerNormalization(epsilon=1e-6, name=f"{name}_ff_ln")(x)
    return x

def build_model():
    input_layer = Input(shape=(HEIGHT, WIDTH, 3), dtype="float32", name="input_layer")
    x = input_layer

    # IMPORTANT FIX: always padding="same" so Conv2D works even if W' becomes 1
    conv_layers_config = {
        "layer1": {"num_conv": 1, "filters":  64, "pool_size": (2, 2)},
        "layer2": {"num_conv": 1, "filters": 128, "pool_size": (2, 2)},
        "layer3": {"num_conv": 2, "filters": 256, "pool_size": (1, 2)},  # shrink width more than height
        "layer4": {"num_conv": 2, "filters": 512, "pool_size": (1, 2)},
        "layer5": {"num_conv": 2, "filters": 128, "pool_size": None},
    }

    for block_name, cfg in conv_layers_config.items():
        for conv_idx in range(1, cfg["num_conv"] + 1):
            x = Conv2D(
                filters=cfg["filters"],
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",                 # <-- FIX HERE
                kernel_initializer="he_uniform",
                name=f"{block_name}_conv{conv_idx}"
            )(x)
            x = BatchNormalization(name=f"{block_name}_bn{conv_idx}")(x)
            x = Activation("relu", name=f"{block_name}_relu{conv_idx}")(x)

        if cfg["pool_size"] is not None:
            x = MaxPooling2D(cfg["pool_size"], name=f"{block_name}_pool")(x)

    # x: (B, H', W', C')
    shape = tf.keras.backend.int_shape(x)
    h, w, c = shape[1], shape[2], shape[3]

    # Vertical OCR: time axis = H'
    x = Reshape(target_shape=(h, w * c), name="reshape_for_transformer")(x)  # (B, T=H', F=W'*C')

    # Transformer encoder (T4-friendly)
    d_model = 256
    num_heads = 4
    ff_dim = 512
    num_layers = 4
    dropout = 0.1

    x = Dense(d_model, name="proj_to_dmodel")(x)
    x = SinusoidalPositionalEncoding(d_model, name="pos_encoding")(x)
    x = Dropout(dropout, name="encoder_input_dropout")(x)

    for i in range(num_layers):
        x = transformer_encoder_block(
            x, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
            name=f"enc{i+1}"
        )

    y_pred = Dense(VOCAB_SIZE + 1, activation="softmax", name="output_layer")(x)
    return Model(inputs=input_layer, outputs=y_pred, name="CNN_Transformer_CTC_Vertical")
