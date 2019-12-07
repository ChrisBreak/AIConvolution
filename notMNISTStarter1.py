#For Google Collab
#try:
#  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
#except Exception:
#  pass

import tensorflow as tf
import numpy as np

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0

print("--Make model--")
model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.005),
                 activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.005),
                 activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.005),
                 activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.005),
                 activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=10, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")
