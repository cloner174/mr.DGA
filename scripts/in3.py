import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

import pandas as pd
data_ = pd.read_csv( 'data/data_from_main.csv' )
X = data_.drop('target', axis=1)
y = data_.values[:,7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

estimator_range = [48, 60]

models = []
scores = []

for n_estimators in estimator_range:
    
    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

# Generate the plot of scores against number of estimators
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize plot
plt.show() 



emo = EmotionRecognizer(feature_shape=feature_shape, output_size=output_size, learning_rate=0.001, num_clusters=5)

emo.build_model()

emo.train(X_train, y_train)







from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
data = pd.read_csv( 'data/data_from_main.csv' , dtype=np.float64)
X_ = data.drop('target', axis=1)
y_ = data.values[:,7]
X = np.asarray(X_, dtype=np.float64)
y = np.asarray(y_, dtype= int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import numpy as np
# Convert y_train from pandas Series to numpy array and reshape it
y = np.array(y_).reshape(-1, 1)
# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
y_train_encoded = encoder.fit_transform(y_train)
model = LogisticRegression(max_iter=1000)
model.fit(X=X_train, y=y_train_encoded)
accuracy = model.score(X_test, y_test_encoded)
print("Accuracy:", accuracy)











import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv( 'data/data_from_main.csv' , dtype=np.float64)
# Assuming X is your feature matrix and y is your target variable
data = pd.DataFrame(X_new, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7'])
data['target'] = y
# Calculate correlation coefficients
correlation_matrix = data.corr()
# Visualize correlation matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()






from sklearn.preprocessing import PolynomialFeatures
# Assuming X is your feature matrix and y is your target variable
# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Concatenate the polynomial features with the original features
X_new = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7']))
# Now you can use X_new for further analysis or modeling







from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)






from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_classifier = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM with RBF Kernel Accuracy:", accuracy)