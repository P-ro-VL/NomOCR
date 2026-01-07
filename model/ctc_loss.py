import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, name='ctc_loss', **kwargs):
        super(CTCLoss, self).__init__(name=name, **kwargs)
        self.loss = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        label_length = tf.cast(y_true != 0, tf.int64)
        label_length = tf.expand_dims(tf.reduce_sum(label_length, axis=-1), axis=1)

        batch_length = tf.cast(tf.shape(y_true)[0], tf.int64)
        pred_length = tf.cast(tf.shape(y_pred)[1], tf.int64)
        pred_length *= tf.ones((batch_length, 1), tf.int64)
        return self.loss(y_true, y_pred, pred_length, label_length)