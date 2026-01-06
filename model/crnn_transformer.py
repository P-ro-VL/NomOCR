import tensorflow as tf
from config import HEIGHT, WIDTH

def transformer_block(x, num_heads=4, ff_dim=512):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)

    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn = tf.keras.layers.Dense(x.shape[-1])(ffn)
    return tf.keras.layers.LayerNormalization()(x + ffn)

def build_model(vocab_size):
    inputs = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, 3))
    x = inputs

    cnn_blocks = [
        (64, (3,3), (2,2)),
        (128, (3,3), (2,2)),
        (256, (3,3), (1,2)),
        (512, (3,3), (1,2)),
    ]

    for i, (filters, kernel, pool) in enumerate(cnn_blocks):
        Conv = tf.keras.layers.Conv2D if i == 0 else tf.keras.layers.SeparableConv2D
        x = Conv(
            filters,
            kernel,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool)(x)

    _, h, w, c = x.shape
    x = tf.keras.layers.Reshape((h, w * c))(x)

    x = transformer_block(x)
    x = transformer_block(x)

    outputs = tf.keras.layers.Dense(vocab_size + 1, activation="softmax")(x)
    return tf.keras.models.Model(inputs, outputs, name="CRNN_Transformer")
