#   #   #                      In the name of God    #  #
#
#cloner174.org@gmail.com
#GitHub.com/cloner174
#
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from scipy.spatial import Delaunay
from keras.models import Sequential
from main import PreProcess
from keras.losses import Huber
Delaunay()
class EmotionRecognizer:
    
    def __init__(self, feature_shape, output_size, learning_rate, num_clusters):
        
        # feature_shape=(48, 48), output_size=7, learning_rate=0.001, num_clusters=5
        
        self.feature_shape = feature_shape
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_clusters = num_clusters
        self.model = None
        self.x_ = None
        self.y_ = None
    
    
    def extract_features_from_delaunay(self):
        """Extract features from facial landmarks using Delaunay triangulation.
        Args:
        image (np.ndarray): Input image.
        Returns:
        np.ndarray: Extracted features.
        """
        
        self.load_facial_landmarks()
        
        features = self.extract_features_from_delaunay_triangulation()
        
        return features
    
    
    def load_facial_landmarks(self):
        """Perform facial landmark detection.
        Args:
        data (np.ndarray): Input image data.
        Returns:
        np.ndarray: Facial landmarks.
        """
        
        obj = PreProcess()
        
        obj.load_data(input_= 'data/sorted_data_pre.csv',index_col_ = 0)
        
        obj.initial_data(need_sort=False)
        
        obj.json_fix()
        
        self.x_, self.y_ = obj.run( n_=self.num_clusters)
    
    
    def extract_features_from_delaunay_triangulation(self):
        """Extract features from facial landmarks using Delaunay triangulation.
        Args:
        facial_landmarks (np.ndarray): Facial landmarks.
        Returns:
        np.ndarray: Extracted features.
        """
        
        triangles = self.calculate_delaunay_triangulation()
        
        features = []
        
        for triangle in triangles:
            feature = np.linalg.norm(triangle[1] - triangle[0]) + \
                       np.linalg.norm(triangle[2] - triangle[1]) + \
                       np.linalg.norm(triangle[0] - triangle[2])
            features.append(feature)
        
        return np.array(features)
    
    
    def calculate_delaunay_triangulation(self):
        """Calculate Delaunay triangulation for facial landmarks.
        Args:
        facial_landmarks (np.ndarray): Facial landmarks.
        Returns:
        list: List of Delaunay triangles as polygon vertices.
        """
        
        delaunay = Delaunay(self.x_)
        
        return delaunay.simplices.reshape(-1, 3, 2)
    
    
    def build_model(self, return_=False):
        """Create the LSTM model for emotion recognition.
        Returns:
        keras.Model: A compiled model.
        """
        # Create a Sequential model
        model = Sequential()
        
        # Add LSTM layer
        model.add(LSTM(32, input_shape=(7, 1), dropout=0.2, return_sequences=False))
        
        # Add densely-connected layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        
        # Add output layer
        model.add(Dense(self.output_size, activation='softmax'))
        
        # Compile the model
        #model.compile(
        #    loss=KLDivergence(),
        #    optimizer=Adam(learning_rate=self.learning_rate),
        #    metrics=['accuracy']
        #)
        model.compile(loss=Huber(), optimizer='adam', metrics=['mae'])  # MAE - Mean Absolute Error
        
        self.model = model
        if return_:
            return model
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, verbose=1, save_model=False):
        """Train the emotion recognition model."""
        # Build the model
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if (X_val is not None and y_val is not None) else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Save the model if required
        if save_model:
            self.model.save("emotion_recognition_model.h5")
        
        return history
#end#