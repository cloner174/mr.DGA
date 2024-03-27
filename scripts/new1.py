from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd



data = pd.read_csv( 'output/dataframe.csv' , dtype=np.float64)

X_ = data.drop('target', axis=1)

y_ = data.values[:,7]


X = np.asarray(X_, dtype=np.float64)

y = np.asarray(y_,  dtype=np.float64)
# Assuming you've imported X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=2)
# Reshape the input data to fit MLPRegressor
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

# Split the data into training and testing sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a pipeline with standard scaler and MLPRegressor
model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=2))

# Fit the model
model.fit(X_train_split, y_train_split)
# Evaluate the model
score = model.score(X_test_split, y_test_split)
print("Model Score:", score)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Model Score:", score)

model.predict(X_test_split)