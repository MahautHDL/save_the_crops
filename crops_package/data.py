import tensorflow as tf
import pandas as pd
import io
import os
import numpy as np
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=y)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=y)
    return train_df, val_df, test_df


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


def prepocessor_img_from_path(file_path):
    # check if the file_path is a URL
    if file_path.startswith('http'):
        response = requests.get(file_path)
        img = tf.io.decode_image(response.content, channels=3)
    else:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
    # Resizing the image
    img = tf.image.resize(img, [224, 224])
    # Add one dimension batch
    img = np.expand_dims(img, axis=0)
    return img


def prepocessor_img(file):
    img = tf.image.decode_image(file, channels=3)
    # Resizing the image
    img = tf.image.resize(img, [224, 224])
    # Add one dimension batch
    img = np.expand_dims(img, axis=0)
    return img
