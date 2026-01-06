import tensorflow as tf
from .preprocess import load_image

def build_dataset(
    image_paths,
    labels,
    char2num,
    max_len,
    batch_size
):
    def encode_label(label):
        label = char2num(tf.strings.unicode_split(label, "UTF-8"))
        return tf.pad(label, [[0, max_len - tf.shape(label)[0]]])

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda x, y: (load_image(x), encode_label(y)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
