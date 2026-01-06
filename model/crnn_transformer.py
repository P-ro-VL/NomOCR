import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization,
    Activation, MaxPooling2D, Reshape, Dense,
    MultiHeadAttention, LayerNormalization, Dropout
)
from tensorflow.keras.models import Model
from config import HEIGHT, WIDTH

def transformer_block(x, num_heads=4, ff_dim=512):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = LayerNormalization()(x + attn)

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(x.shape[-1])(ffn)
    return LayerNormalization()(x + ffn)

def build_model(vocab_size):
    inputs = Input(shape=(HEIGHT, WIDTH, 3))
    x = inputs

    cnn_blocks = [
        (64, (3,3), (2,2)),
        (128, (3,3), (2,2)),
        (256, (3,3), (1,2)),
        (512, (3,3), (1,2)),
    ]

    for i, (filters, kernel, pool) in enumerate(cnn_blocks):
        Conv = Conv2D if i == 0 else SeparableConv2D
        x = Conv(
            filters,
            kernel,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform"
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool)(x)

    _, h, w, c = x.shape
    x = Reshape((h, w * c))(x)

    x = transformer_block(x)
    x = transformer_block(x)

    outputs = Dense(vocab_size + 1, activation="softmax")(x)
    return Model(inputs, outputs, name="CRNN_Transformer")
