import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        batch = tf.shape(y_true)[0]
        input_len = tf.shape(y_pred)[1] * tf.ones((batch, 1), dtype="int32")
        label_len = tf.math.count_nonzero(y_true, axis=1, keepdims=True)

        return tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_len, label_len
        )
