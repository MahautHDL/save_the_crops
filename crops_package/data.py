import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.model_selection import train_test_split

def split_data(df, size_to_throw=0.9):
    resized_df, to_throw = train_test_split(df, test_size=size_to_throw, random_state=42)
    train_df, test_df = train_test_split(resized_df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

    return train_df, val_df, test_df


# def preprocessor():
#     tf.keras.preprocessing.image_dataset_from_directory(
#         directory=os.environ.get("LOCAL_PATH"),
#         labels='inferred',
#         image_size=(224,224),
#         # class_names=
#         label_mode="categorical",
#         batch_size=32
#     )

def preprocessor_df(df):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    return generator

def preprocessor_test(df):
    datagen = ImageDataGenerator(
        rescale=1./255,

    )
    generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    return generator


def prepocessor_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)  # convert to tensor and define 3 colors
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    return img
