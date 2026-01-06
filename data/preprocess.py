import tensorflow as tf
from config import HEIGHT, WIDTH

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (HEIGHT, WIDTH), preserve_aspect_ratio=True)
    img = tf.cast(img, tf.float32) / 255.0
    return img
