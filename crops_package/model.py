import tensorflow as tf
import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping

# Create tensorboard callback (functionized because need to create a new one for each model)
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

import tensorflow as tf
import datetime
import os

def fit_and_save_model(model, X, validation_data, crop, patience=5, epochs = 100):
    """
    Will fit and save the model.
    Input:
        * model
        * X: preprocessed training data
        * validation_data: preprocessed val_data
        * crop
    Optional input:
        * patience (5)
        * epochs (100)
    Output:
        * history of model
    Further actions:
        * model is saved in folder models as name_of_crop-model.keras
        * tf callbacks are saved in callbacks folder as name_of_crop_model
    """
    # Early stopping
    es = EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True)

    # Fitting the model
    history = model.fit(X,
                    validation_data= validation_data,
                    epochs = epochs,
                    batch_size = 32,
                    callbacks = [es,
                                create_tensorboard_callback("../../callbacks/", # save experiment logs here
                                experiment_name=f"{crop}_model")] # name of log files]
                                 )

    # Save model
    model.save(f'../../models/saved_models/{crop}-model.keras')
    return history

from tensorflow.keras import layers, models

def initialize_model(nr_of_classes):
    pretrained_model = tf.keras.applications.ResNet101V2(
            include_top=False ,
            input_shape=[224, 224, 3]
        )
    pretrained_model.trainable = False

    model = tf.keras.Sequential([
            tf.keras.layers.Lambda(tf.keras.applications.resnet_v2.preprocess_input),

            pretrained_model,

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(nr_of_classes, activation='softmax')
        ])

    ### Model compilation

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  run_eagerly=True,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

def initialize_model_test(nr_of_classes):
    model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
            tf.keras.layers.Dense(nr_of_classes, activation="relu"),
            # tf.keras.layers.Dense(nr_of_classes, activation='softmax')
        ])

    ### Model compilation

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  run_eagerly=True,
                  metrics=['accuracy'])

    return model
