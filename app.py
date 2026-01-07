import os
import zipfile
import beam
from beam import Sandbox, Image, Volume, endpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import *
from model.crnn_transformer import build_model
from model.ctc_loss import CTCLoss
from data.dataset import build_dataset


DATASET_DIR = "./dataset"
ZIP_PATH = os.path.join(DATASET_DIR, "dataset.zip")
EXTRACT_DIR = os.path.join(DATASET_DIR, "data")


def unzip_if_needed():
    if os.path.exists(os.path.join(EXTRACT_DIR, "labels.txt")):
        print("Dataset already extracted, skipping unzip")
        return

    print("Extracting dataset zip...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset extracted")


@endpoint(
    name="crnn-transformer-sandbox",
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    gpu_count=1,
    image=Image(
        python_packages=[
            "tensorflow[and-cuda]",
            "scikit-learn",
        ]
    ),
    volumes=[
        Volume(name="dataset", mount_path="./dataset"),
        Volume(name="checkpoints", mount_path=CHECKPOINT_DIR),
    ],
)
def train():
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices())
    print("Starting training...")

    unzip_if_needed()

    image_root = os.path.join(EXTRACT_DIR, "images")
    label_file = os.path.join(EXTRACT_DIR, "labels.txt")

    # Load labels
    with open(label_file, encoding="utf-8") as f:
        lines = [l.strip().split("\t") for l in f if l.strip()]

    image_paths = [os.path.join(image_root, x[0]) for x in lines]
    labels = [x[1] for x in lines]

    print(f"Loaded {len(labels)} samples")

    vocab = sorted(set("".join(labels)))
    char2num = tf.keras.layers.StringLookup(
        vocabulary=vocab,
        mask_token=PADDING_TOKEN
    )

    max_len = max(len(l) for l in labels)

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        image_paths,
        labels,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    train_ds = build_dataset(
        train_imgs, train_lbls, char2num, max_len, BATCH_SIZE
    )
    val_ds = build_dataset(
        val_imgs, val_lbls, char2num, max_len, BATCH_SIZE
    )

    print("Building model...")
    model = build_model(len(vocab))
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
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Training completed!")




# if __name__ == "__main__":
#     sandbox = Sandbox(
#         name="crnn-transformer-sandbox",
#         cpu=4,
#         memory="32Gi",
#         gpu="A10G",
#         gpu_count=1,
#         image=Image(
#             python_packages=[
#                 "tensorflow[and-cuda]",
#                 "scikit-learn",
#             ]
#         ),
#         volumes=[
#             Volume(name="dataset", mount_path="./dataset"),
#             Volume(name="checkpoints", mount_path=CHECKPOINT_DIR),
#         ],
#     )

#     sb = sandbox.create()

#     print(os.system("nvidia-smi"))
    
#     sb.process.run_code(train())
#     print("Done")

#     sb.terminate()