from tensorflow import keras
from tensorflow.summary import create_plot_model_to_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class EmotionRecognizer:
  def __init__(self, feature_shape, output_size, learning_rate=0.001):
    # Define model architecture
    self.model = self.build_model(feature_shape, output_size)
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
  def build_model(self, feature_shape, output_size):
    # CNN block for capturing spatial features
    cnn_block = keras.Sequential([
      keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=feature_shape),
      keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])
    # LSTM block for handling sequential information (if applicable)
    lstm_block = keras.layers.LSTM(units=64, return_sequences=True)  # Adjust units as needed
    # Combine CNN and LSTM outputs (if using LSTM)
    combined_layer = keras.layers.concatenate([cnn_block(inputs=keras.Input(shape=feature_shape)), lstm_block(inputs=cnn_block(inputs=keras.Input(shape=feature_shape)))])  # Modify for single input if not using LSTM
    # Final classification layers
    output_layer = keras.Sequential([
      combined_layer,  # Replace with cnn_block output if not using LSTM
      keras.layers.Flatten(),
      keras.layers.Dense(output_size, activation='softmax')
    ])
    return output_layer
  def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10):
    """Trains the model on provided features and labels with optional Early Stopping.
    Args:
      X_train (np.ndarray): Training features.
      y_train (np.ndarray): Training labels.
      X_val (np.ndarray, optional): Validation features (for Early Stopping). Defaults to None.
      y_val (np.ndarray, optional): Validation labels (for Early Stopping). Defaults to None.
      epochs (int, optional): Number of training epochs. Defaults to 10.
    """
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)  # Stop training if validation accuracy doesn't improve for 5 epochs
    if X_val is not None and y_val is not None:
      # Train with Early Stopping if validation data is provided
      self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    else:
      # Train without Early Stopping if no validation data is provided
      self.model.fit(X_train, y_train, epochs=epochs)
  def predict(self, X_test):
    return self.model.predict(X_test)
  def visualize_model(self, filepath="model.png"):
    """Visualizes the model architecture and saves it to a file.
    Args:
      filepath (str, optional): Path to save the visualization image. Defaults to "model.png".
    """
    create_plot_model_to_file(self.model, to_file=filepath, show_shapes=True, dpi=64)





data = pd.read_csv( 'data/data_from_main.csv' )

print(data.shape,
      type(data.iloc[0,0]),
      type(data.iloc[5,5]))

X_ = data.drop('target', axis=1)
y_ = data.values[:,7]

X = np.asarray(X_)
y = np.asarray(y_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)


# Example usage
# Assuming your features are reshaped for CNN input (e.g., 2D tensor)
feature_shape = (161491, 7)  # Replace with actual feature dimensions
output_size = 1 # Number of emotion classes

# Create the emotion recognizer
recognizer = EmotionRecognizer(feature_shape, output_size)

# Train the model
recognizer.train(X_train, y_train, epochs=100)

# Use the trained model for prediction on new data
X_new = ...  # Reshaped features for new data
predictions = recognizer.predict(X_new)


# Example usage
# ... (rest of the code)

# Visualize the model architecture
recognizer.visualize_model()
