import tensorflow as tf
import os

def preprocessor():
    tf.keras.preprocessing.image_dataset_from_directory(
        directory=os.environ.get("LOCAL_PATH"),
        labels='inferred',
        image_size=(224,224),
        # class_names=
        label_mode="categorical",
        batch_size=32
    )
