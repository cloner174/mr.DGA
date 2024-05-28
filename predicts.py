#In the name of God#
import time
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad


class SimpleNetwork:
    
    def __init__(self, number_of_input_features, number_of_output_classes):
        
        self.model = None
        self.num_feature = number_of_input_features
        self.output_dim = number_of_output_classes
    
    def create_model(self,
                     activation_for_input_layer = 'relu', activation_for_hidden_layer= 'relu' ) :
        # structure
        model = Sequential([
            Dense(128, activation = activation_for_input_layer, input_shape=(self.num_feature,)),# Input layer
            Dropout(0.2),  # dropout layer for regularization
            Dense(128, activation = activation_for_hidden_layer),  # hidden layer
            Dropout(0.2),
            Dense(self.output_dim, activation='softmax')  # Output layer for classification
            ])
        self.model = model
    
    def compile_model(self, 
                      optimizer = 'adam', 
                      loss_function = 'sparse_categorical_crossentropy',
                      metric = 'accuracy') :
        
        model = self.model if self.model is not None else self.create_model()
        model.compile(optimizer = optimizer,
                      loss = loss_function,  # also good :'categorical_crossentropy' if one-hot
                      metrics = [metric] )
        return model
    


class AdvanceNetwork:
    
    def __init__(self, input_dim, output_dim) :
        
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def __hidden_layer__(self, dropout):
        
        return [ 
                Dense(64, activation = self.activation_for_hidden_layer),
                BatchNormalization(),
                Dropout(dropout)
            ]
    
    def __input_layer__(self, dropout):
        
        return [ 
                Dense(128, activation = self.activation_for_input_layer, input_shape=(self.input_dim,)),
                BatchNormalization(),  # Normalize the activations of the previous layer at each batch
                Dropout(dropout),  # to prevent overfitting
                Dense(128, activation = self.activation_for_input_layer),
                BatchNormalization(),
                Dropout(dropout) 
            ]
    
    def __output_layer__(self, dropout) :
        
        return [
                Dense(32, activation = self.activation_for_hidden_layer),
                BatchNormalization(),
                Dropout(dropout),
                Dense(self.output_dim, activation = self.activation_for_output_layer)
            ]
    
    def create_model(self, 
                     activation_for_input_layer = 'relu',
                     activation_for_hidden_layer= 'relu' ,
                     activation_for_output_layer = 'softmax',
                     num_layers = 4, 
                     dropout = 0.2):
        
        self.activation_for_hidden_layer = activation_for_hidden_layer
        self.activation_for_input_layer = activation_for_input_layer
        self.activation_for_output_layer = activation_for_output_layer
        if num_layers > 10 :
            dropout = 0.1
        if num_layers < 4 :
            num_layers = 4
        out_drop = dropout if dropout < 0.1 else 0.1
        num_hidden = 5 - num_layers
        model_layers = []
        in_layer = self.__input_layer__(dropout)
        hide_layer = self.__hidden_layer__(dropout)
        out_layer = self.__output_layer__(out_drop)
        model_layers.extend(in_layer)
        for i in range(num_hidden):
            model_layers.extend(hide_layer)
        model_layers.extend(out_layer)
        self.model = Sequential(model_layers)
    
    def compile_model(self, 
                      optimizer = 'adam', 
                      loss_function = 'sparse_categorical_crossentropy',
                      metric = 'accuracy') :
        
        if self.model is None:
            try:
                self.create_model()
            except:
                print("Unable to detect the model")
                raise
        model = self.model
        model.compile(optimizer = optimizer,
                      loss = loss_function,  # also good :'categorical_crossentropy' if one-hot
                      metrics = [metric] )
        
        return model



def train(
          model,  
          X_train, y_train, 
          X_val =None, y_val=None, validation_split = None,
          epochs = 10, 
          batch_size=16, 
          verbose = 1,
          early_stopping = None, 
          model_checkpoint = None ):
    
    validation = None
    validation_split = validation_split if validation_split is not None else 0.1
    if X_val is None and y_val is None :
        warnings.warn(f" Nither The X_val nor y_val was passed . \n  Start To train using default setting-> validation_split = {validation_split}")
        time.sleep(2)
    elif X_val is not None and y_val is not None:
        warnings.warn(" Both The validation data was Detected . \n Start To train using Validation Data ")
        time.sleep(2)
        validation = (X_val, y_val)
    elif X_val is None or y_val is None:
        warnings.warn(f" One of The validation data 'X or y' are None \n  Start To train using default setting-> validation_split = {validation_split}")
        time.sleep(2)
    elif validation is None:
        warnings.warn(f" Start To train using default setting-> validation_split = {validation_split} ")
        time.sleep(2)
    else:
        warnings.warn(" 'X' NOT Detected or 'y' NOT Detected -> None ->  validation split float NOT Detected . \n Start To train WITHOUT validation ")
        time.sleep(2)
        raise OSError("No Way")
    if model_checkpoint is None or early_stopping is None :
        if validation is None:
            history = model.fit(X_train, y_train,epochs=epochs,batch_size=batch_size, validation_split=validation_split,verbose=verbose )
        else:
            history = model.fit(X_train, y_train,epochs=epochs,batch_size=batch_size, validation_split=validation_split,verbose=verbose )
    elif model_checkpoint is not None and early_stopping is not None :
        if validation is None:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, model_checkpoint],
                verbose=verbose)     
        else:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data = validation,
                callbacks=[early_stopping, model_checkpoint],
                verbose=verbose)
    else:
        warnings.warn(" 'model_checkpoint' NOT Detected or 'early_stopping' NOT Detected \n Start To train Using Default Mode : Simple Training")
        time.sleep(2)
        history = model.fit(X_train, y_train,epochs=epochs, batch_size=batch_size)
    
    return model, history



def select_optimizer( name = 'adam', learning_rate = 0.001, param1 = 0.9, param2 = 0.999, param3 = 1e-08, param4 = 0.0):
    
    if name.lower() == 'adam':
        optim = Adam(learning_rate=learning_rate, beta_1=param1, beta_2=param2, epsilon=param3, decay=param4)
    elif name.lower() == 'sgd':
        optim = SGD(learning_rate=learning_rate, momentum=param1)
    elif name.lower() == 'rmsprop':
        optim = RMSprop(learning_rate=learning_rate, rho=param1)
    elif name.lower() == 'adagrad':
        optim = Adagrad(learning_rate=learning_rate)
    else:
        raise AssertionError(" Invalid name of optimizer or unsupported ! valid choices are : 'adam','sgd','rmsprop','adagrad' ")
    return optim



class Evaluation :
    
    
    def __init__(self, model, history, scaler = None) :
        
        self.model = model
        self.history = history
        self.scaler = scaler
        self.test_accuracy = None
        self.test_loss = None
    
    def run(self,X_test,  y_test, verbose=2):
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=verbose)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
    
    def plot_results(self, type = 'default'):
        
        if type.lower() == 'default':
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        else:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        
        if self.test_accuracy is not None :
            plt.plot(self.test_accuracy)
            plt.title('Test Accuracy')
            plt.xlabel('')
            plt.ylabel('')
            plt.show()
        
        if self.test_loss is not None:
            plt.plot(self.test_accuracy)
            plt.title('Test Loss')
            plt.xlabel('')
            plt.ylabel('')
            plt.show()            




"""
model = Sequential([
    Dense(128, input_shape=(10,)),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),

    Dense(128),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])
model = Sequential([
    Dense(128, activation='relu', , input_shape=(10,)),
    Dropout(0.2),

    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
def build_model(optimizer='adam'):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(10,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=build_model, verbose=0)
optimizers = ['rmsprop', 'adam']
param_grid = dict(optimizer=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
def predict_from_csv(model_path, data_folder, csv_filename):
    model = load_model(model_path)
    csv_path = f"{data_folder}/{csv_filename}"
    data = pd.read_csv(csv_path)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    predictions = model.predict(data)
    # predictions = (predictions > 0.5).astype(int)
    return predictions
model_path = 'path_to_your_model.h5'
data_folder = 'path_to_your_data_folder'
csv_filename = 'data.csv'
predictions = predict_from_csv(model_path, data_folder, csv_filename)
print(predictions)
predictions = model.predict(data)
predicted_classes = np.argmax(predictions, axis=1)
predictions = model.predict(data)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
predictions = model.predict(X_new)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes) 
"""
#cloner174#2024