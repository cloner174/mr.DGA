import numpy as np
import pandas as pd



data = pd.read_csv( 'output/dataframe.csv' , dtype=np.float64)

X_ = data.drop('target', axis=1)

y_ = data.values[:,7]


X = np.asarray(X_, dtype=np.float64)

y = np.asarray(y_,  dtype=int)
# Assuming you've imported X_train and y_train

X = X.reshape(35887, 6, 7)
y = y.reshape(35887,6)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Assuming you've imported X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)



# Assuming you've imported X_train and y_train

# Split the data into training and testing sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and LSTM model
model = make_pipeline(StandardScaler(), Sequential([
    LSTM(units=100, input_shape=(X_train.shape[0], X_train.shape[1])),
    Dense(units=6)  # Output layer with 6 neurons since y_train has shape (a, 6)
]))

# Compile the model
model.steps[-1][1].compile(optimizer='adam', loss='mse')  # Accessing the Keras model within the pipeline and compiling it
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test_split)
score = r2_score(y_test_split, y_pred)
print("Model Score:", score)


X_t = StandardScaler()
X_train = X_t.X_train