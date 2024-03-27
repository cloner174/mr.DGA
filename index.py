import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv( 'data/data_from_main.csv' , dtype=np.float64)

X_ = data.drop('target', axis=1)

y_ = data.values[:,7]


X = np.asarray(X_, dtype=np.float64)
y = np.asarray(y_, dtype= int)


# Assuming X is your feature matrix and y is your target variable

#             UseFull Other Time  #
#data = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7'])
#data['target'] = y
#             UseFull Other Time  #


# Calculate correlation coefficients
correlation_matrix = data.corr()

# Visualize correlation matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()


logreg_model = LogisticRegression(max_iter=1000)

rfe = RFE(estimator=logreg_model, n_features_to_select=5)

rfe.fit(X, y)

print("Selected Features:", rfe.support_)



data = pd.read_csv( 'data/data_from_main.csv' , dtype=np.float64)

X_ = data.drop(['target', 'Mode','Variance'], axis=1)  # Base on Above

X = np.asarray(X_, dtype=np.float64)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", accuracy)














# OLDERRRRRRRRRRRRR ANDDDDDDDD TRASHHHHHHHHHHHHHHH

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

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=32,shuffle=True)



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_train_reshaped = np.expand_dims(X_train, axis=-1)
# Now X_train_reshaped should have a shape of (172257, 7, 1), which matches the expected input shape
# Determine the shape of each feature vector
feature_shape = X_train.shape[1:]  # (7,)
# Determine the number of unique classes
output_size = len(np.unique(y_train))