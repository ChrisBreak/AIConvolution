try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd

print("--Get data--")

trainPath = "heart_train.csv"
testPath = "heart_test.csv"

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

!head {trainPath}
!head {testPath}

LABEL_COLUMN = 'chd'
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=50,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
  return dataset

COLUMNS = ['sbp','tobacco','ldl','adiposity','famhist', 'typea','obesity','alcohol','age','chd']

rawTrainData = get_dataset(trainPath, select_columns = COLUMNS)
rawTestData = get_dataset(testPath, select_columns = COLUMNS)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

show_batch(rawTrainData)

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

NUMERIC_FEATURES = ['sbp','tobacco','ldl','adiposity', 'typea','obesity','alcohol','age']

packed_train_data = rawTrainData.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = rawTestData.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)

trainBatch, labels_batch = next(iter(packed_train_data))
testBatch, labels_batch = next(iter(packed_test_data))

descript = pd.read_csv(trainPath)[NUMERIC_FEATURES].describe()
descript

MEAN = np.array(descript.T['mean'])
STD = np.array(descript.T['std'])

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

trainBatch['numeric']
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(trainBatch).numpy()

CATEGORIES = {'famhist': ['Present', 'Absent']}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

categorical_columns

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(trainBatch).numpy()[0])

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

print(preprocessing_layer(trainBatch).numpy()[0])

print("--Make model--")

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

print("--Fit model--")
model.fit(train_data, epochs=20, verbose=2)

print("--Evaluate model--")
test_loss, test_accuracy = model.evaluate(test_data, verbose=2)

print(f"Model Loss:    {test_loss:.2f}")
print(f"Model Accuray: {test_accuracy*100:.1f}%")

predictions = model.predict(test_data)

# Show some predictions
for prediction, isCHD in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Prediction for CHD: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("Has CHD" if bool(isCHD) else "Does not have CHD"))
