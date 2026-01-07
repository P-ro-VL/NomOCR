import os
import zipfile
import beam
from beam import Sandbox, Image, Volume
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import *
from model.crnn_transformer import build_model
from model.ctc_loss import CTCLoss
from data.preprocess import DataImporter, DataHandler

def train():
    print("Starting training...")

    image_root = os.path.join(DATASET_DIR, "images")
    label_file = os.path.join(DATASET_DIR, "labels.txt")

    dataset = DataImporter(DATASET_DIR, label_file, min_length=1)
    data_handler = DataHandler(dataset, img_size=(HEIGHT, WIDTH), padding_char=PADDING_TOKEN)

    VOCAB_SIZE = len(data_handler.char2num.get_vocabulary())
    print(f"Vocabulary size: {VOCAB_SIZE}")

    train_idxs = list(range(dataset.size - NUM_VALIDATE))
    valid_idxs = list(range(train_idxs[-1] + 1, dataset.size))

    train_tf_dataset = data_handler.prepare_tf_dataset(train_idxs, BATCH_SIZE)
    valid_tf_dataset = data_handler.prepare_tf_dataset(valid_idxs, BATCH_SIZE)
    
    print("Building model...")
    model = build_model()
    model.summary(line_length=110)
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(LEARNING_RATE),
        loss=CTCLoss()
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_ckpt = os.path.join(
        CHECKPOINT_DIR, "crnn_transformer.best.weights.h5"
    )
    last_ckpt = os.path.join(
        CHECKPOINT_DIR, "crnn_transformer.last.weights.h5"
    )

    if os.path.exists(best_ckpt):
        print("Loading best checkpoint...")
        model.load_weights(best_ckpt)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_ckpt,
            save_best_only=False,
            save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    print("Starting model training...")
    history = model.fit(
        train_tf_dataset,
        validation_data=valid_tf_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Training completed!")


if __name__ == "__main__":
    sandbox = Sandbox(
        name="crnn-transformer-sandbox",
        cpu=4,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_packages=[
                "tensorflow",
                "scikit-learn",
            ]
        ),
        volumes=[
            Volume(name="dataset", mount_path="./dataset"),
            Volume(name="checkpoints", mount_path=CHECKPOINT_DIR),
        ],
    )

    sb = sandbox.create()
    
    sb.process.run_code(train())
    print("Done")

    sb.terminate()