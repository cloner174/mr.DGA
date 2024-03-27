import pandas as pd
import numpy as np
from tensorflow import keras
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
#from scipy import stats
#import re
from sk import EmotionRecognizer



data = pd.read_csv( 'data/data_from_main.csv' , dtype=np.float64)


print(type(data),
      type(data.iloc[0,0]),
      type(data.iloc[5,5]))


X_ = data.drop('target', axis=1)

y_ = data.values[:,7]


X = np.asarray(X_, dtype=np.float64)

y = np.asarray(y_,  dtype=np.float64)

x = list(zip(X[:],y))

x = np.array(x)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2)



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_train_reshaped = np.expand_dims(X_train, axis=-1)

# Now X_train_reshaped should have a shape of (172257, 7, 1), which matches the expected input shape

# Determine the shape of each feature vector
feature_shape = X_train.shape[1:]  # (7,)

# Determine the number of unique classes
output_size = len(np.unique(y_train))

emo = EmotionRecognizer(feature_shape=feature_shape, output_size=output_size, learning_rate=0.001, num_clusters=5)

emo.build_model()

emo.train(X_train, y_train)