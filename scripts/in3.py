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