import pandas as pd
import numpy as np
#from scipy import stats
#import re

data = pd.read_csv( 'data/data_from_main.csv' )

print(type(data),
      type(data.iloc[0,0]),
      type(data.iloc[5,5]))

X_ = data.drop('target', axis=1)
y_ = data.values[:,7]

X = np.asarray(X_)
y = np.asarray(y_)


from sklearn. import spli