import tensorflow as tf
import pandas as pd
import io
import os
import numpy as np
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def check_for_errors_and_create_excel():
    """
    Will check if pictures are not corrupted.
    Then it will create and save a csv with the filename and class of the non-corrupted pictures.
    Output is a dataframe from this csv.
    """
    data_link = os.environ.get("LOCAL_PATH")

    data = pd.DataFrame(columns=["filename", "class"])

    crops = os.listdir(data_link)

    for i in range(0,len(crops)):
        plant_disease_name = crops[i]
        os.chdir(f'{data_link}{plant_disease_name}')

        filenames = os.listdir(f"{data_link}{plant_disease_name}")

        for index, filename in enumerate(filenames):
            if filename.endswith(".jpg"):

                try:
                    plt.imread(filename)
                    data.loc[len(data)] = [f'{plant_disease_name}/{filename}', plant_disease_name]

                except:
                    pass
            else:
                pass

    data.to_csv(f'{os.environ.get("DATA_PATH")}data.csv', index=False)

    return data


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['class'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['class'])
    return train_df, val_df, test_df


def split_and_reduce_data(df, size_to_throw=0.001):
    """
    Will split data into train_df, val_df and test_df.
    size_to_throw is the amount of data not to use (between 0.001 and 1)
    """
    resized_df, to_throw = train_test_split(df, test_size=size_to_throw, random_state=42, stratify=df['class'])
    train_df, test_df = train_test_split(resized_df, test_size=0.15, random_state=42, stratify=resized_df['class'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['class'])
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
        target_size=(128, 128),
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
        target_size=(128, 128),
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
    img = tf.image.resize(img, [128, 128])
    # Add one dimension batch
    img = np.expand_dims(img, axis=0)
    return img


def prepocessor_img(file):
    img = tf.image.decode_image(file, channels=3)
    # Resizing the image
    img = tf.image.resize(img, [128, 128])
    # Add one dimension batch
    img = np.expand_dims(img, axis=0)
    return img
