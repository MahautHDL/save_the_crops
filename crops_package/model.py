import tensorflow as tf
import datetime
import os

from tensorflow.keras.callbacks import EarlyStopping

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

def fit_and_save_model(model, X, validation_data, model_name, split=0.3, epochs=100):
    es = EarlyStopping(patience = 5, verbose = 2, restore_best_weights=True)

    history = model.fit(X,
                        validation_data = validation_data,
                        epochs = epochs,
                        callbacks=[es, create_tensorboard_callback(dir_name=os.environ.get("CALLBACK_PATH"), # save experiment logs here
                                                                         experiment_name=model_name)] # name of log files
    )

    model.save(f'./models/{model_name}')
    return history
