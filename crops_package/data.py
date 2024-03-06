import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.model_selection import train_test_split

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
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
