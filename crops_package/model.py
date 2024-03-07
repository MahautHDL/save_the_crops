import tensorflow as tf
import datetime
import os

from tensorflow.keras.callbacks import EarlyStopping


# Create tensorboard callback (functionized because need to create a new one for each model)
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# fitting a model

def fit_model(model, X, y, name_callback, split=0.3, epochs=100):
    es = EarlyStopping(patience = 5, verbose = 2)

    history = model.fit(X,
                        y,
                        validation_split = split,
                        callbacks = [es],
                        epochs = epochs,
                        callbacks=[create_tensorboard_callback(dir_name=os.environ.get("CALLBACK_PATH"), # save experiment logs here
                                                                         experiment_name=name_callback)] # name of log files
    )

    return history
