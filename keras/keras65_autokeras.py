# 64번으로 테스트
import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence

x_train = np.load('./data/keras63_train_x.npy')
x_test = np.load('./data/keras63_test_x.npy')
y_train = np.load('./data/keras63_train_y.npy')
y_test = np.load('./data/keras63_test_y.npy')

print(x_train.shape)
print(y_train.shape)
print(y_test[:3])

# Initialize the image classifier.
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

# Feed the image classifier with trainin data.
clf.fit(x_train, y_train, epochs=50)

# Predict with the best model.
predict_y = clf.predict(x_test)
predict_y(predict_y)

# Evaluate the best model
predict_y(clf.evaluate(x_test, y_test))
# clf.summary()