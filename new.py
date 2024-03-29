import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SKlearn:
    
    def __init__(self) :
        
        self.X = None
        self.y = None
        self.corr_ = None
        self.logist_model = None
        self.best_cols = None
        self.data_columns = None
        self.standard_X = None
        self.MinMax_X = None
        self.Manual_X = None
    
    
    def load_data(self, data_path) :
        
        
        self.data = data_path
        self.data = pd.read_csv(self.data)
    
    
    def initial_data(self, target_col_name = 'target', return_ = False, also_return_data = False) :
        
        self.tar_col_name = target_col_name
        
        self.data_columns = list(self.data.columns)
        
        self.X = self.data.drop(self.tar_col_name, axis=1)
        
        self.y = self.data.loc[:,self.tar_col_name]
        
        if return_:
            if also_return_data:
                return self.data, self.X, self.y
            else:
                return self.X, self.y
    
    
    def make_array(self, reType_All = False, return_ = True) :
        
        self.initial_data()
        
        X = np.asarray(self.X)
        
        y = np.asarray(self.y)
        
        if reType_All :
            self.X = X
            self.y = y
            
            if return_ == False:
                return
        
        return X, y
    
    
    def cal_corr(self, show_ = True, return_ = False) :
        
        self.corr_ = self.data.corr()
        
        if show_ == False:
            if return_:
                return self.corr_
            else:
                return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.corr_, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
        plt.title('Correlation Matrix')
        plt.show()
    
    def scaling(self, all_ = False, return_ = False) :
        
        stand_scale = StandardScaler()
        relu_scale =  MinMaxScaler()        
        
        if all_:
            self.standard_X = stand_scale.fit_transform(self.X)
            self.MinMax_X = relu_scale.fit_transform(self.X)
            self.data = pd.DataFrame(self.standard_X)
            self.data['target'] = self.y
            
            if return_:
                return self.data
            else:
                return
        
        material, _ = self.make_array()
        
        self.standard_X = stand_scale.fit_transform(material)
        self.MinMax_X = relu_scale.fit_transform(material)
        
        if return_:
            return self.standard_X, self.MinMax_X
    
    
    def manual_scale(self) :
        
        self.Manual_X, _ = self.make_array()
        for i in range(self.Manual_X.shape[0]) :
            for j in range(self.Manual_X.shape[1]) :
                
                
                temp = self.Manual_X[i,j]
                if temp > 0.49 :
                    self.Manual_X[i,j] = 0
                else:
                    self.Manual_X[i,j] = 1
    
    
    def get_best_features(self,feturs_to_slct = 5,
                          standard_X = False,
                          MinMax_X = False ,
                          manual_X_scaled = False,
                          Just_for_corr = False, 
                          silently = True, 
                          return_ = False) :
        
        self.logist_model = LogisticRegression(C=0.1, solver='sag')
        
        gbf = RFE( estimator = self.logist_model, 
                  n_features_to_select = feturs_to_slct)
        
        if standard_X:
            self.scaling()
            X = self.standard_X
            _, y = self.make_array()
        elif MinMax_X:
            self.scaling()
            X = self.MinMax_X
            _, y = self.make_array()   
        elif manual_X_scaled:
            self.manual_scale()
            X = self.Manual_X
            _, y = self.make_array()
        else:
            X, y = self.make_array()
        
        data_x = pd.DataFrame(self.standard_X)
        data_y = pd.Series(self.y)
        data = data_x.assign( target = data_y )
        if standard_X:
            data_x = pd.DataFrame(self.standard_X)
            data_y = pd.Series(self.y)
            data = data_x.assign( target = data_y )
            if Just_for_corr:
                self.data = data
                self.initial_data()
                return
            else:
                self.data = data
                self.initial_data()
        
        if MinMax_X:
            data_x = pd.DataFrame(self.MinMax_X)
            data_y = pd.Series(self.y)
            data = data_x.assign( target = data_y )
            if Just_for_corr:
                self.data = data
                self.initial_data()
                return
            else:
                self.data = data
                self.initial_data()
        
        if manual_X_scaled:
            data_x = pd.DataFrame(self.Manual_X)
            data_y = pd.Series(self.y)
            data = data_x.assign( target = data_y )
            if Just_for_corr:
                self.data = data
                self.initial_data()
                return
            else:
                self.data = data
                self.initial_data()                 
        
        gbf.fit(X, y)
        
        self.best_cols = list(gbf.support_)
        temp_del_invalid_col = []
        for i in range( len( self.best_cols )) :
            
            temp = self.best_cols[i]
            if temp != True:
                temp_del_invalid_col.append( self.data_columns[i] )
        
                
        self.data = self.data.drop(temp_del_invalid_col, axis=1)
        self.initial_data()
        
        if silently == False:
            print("Selected Features:", gbf.support_)
        
        if return_:
            return self.data
    
    
    def models(self, to_save_ = 'output/RandomForestClassifier.csv',
               test_size_ = 0.2, n_estimators_ = 1000, ) :
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size_)
        
        rf_classifier = RandomForestClassifier( n_estimators = n_estimators_)
        
        rf_classifier.fit(X_train, y_train)
        
        y_pred = rf_classifier.predict(X_test)
        
        y_test = pd.DataFrame(y_test)
        to_save = y_test.assign( y_pred = y_pred )
        to_save.to_csv(to_save_)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Random Forest Accuracy:", accuracy)