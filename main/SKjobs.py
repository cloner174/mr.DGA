#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
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
from main.stats import NameHelper


class SKlearn :
    

    def __init__(self) :
        
        self.data = None
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


    def load_data_from_PreProcess(self) :
        
        from main import PreProcess
        
        pre_process = PreProcess()
        pre_process.load_data()
        pre_process.initial_data()
        self.data = pre_process.fix_data(return_=True)    


    def initial_data(self, target_col_name = 'target', return_Xy = False, also_return_data = False) :
        
        self.tar_col_name = target_col_name
        
        self.data_columns = list(self.data.columns)
        
        self.X = self.data.drop(self.tar_col_name, axis=1)
        
        self.y = self.data.loc[:,self.tar_col_name]
        
        if return_Xy:
            if also_return_data:
                return self.X, self.y, self.data
            else:
                return self.X, self.y   


    def make_array(self, reType_All = False, return_ = True) :
        
        self.initial_data(self.tar_col_name)
        
        X = np.asarray(self.X)
        
        y = np.asarray(self.y)
        
        if reType_All :
            self.X = X
            self.y = y
            
            if return_ == False:
                return
        
        return X, y


    def cal_corr(self,
                 show_ = True,
                 return_ = False,
                 save_ = False,
                 save_where = 'data/Faze1/output/Pics/cal_corr') :
        
        self.corr_ = self.data.corr()
        
        if show_ == False:
            if return_:
                return self.corr_
            else:
                return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.corr_, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
        plt.title('Correlation Matrix')
        
        if save_:
            id_ = NameHelper()
            plt.savefig(f"{save_where}/{id_}.jpg")
        plt.show()
    
    
    def scaling(self, all_ = False, return_ = False) :
        
        stand_scale = StandardScaler()
        relu_scale =  MinMaxScaler()        
        
        if all_:
            self.standard_X = stand_scale.fit_transform(self.X)
            self.MinMax_X = relu_scale.fit_transform(self.X)
            self.data_stan = pd.DataFrame(self.standard_X)
            self.data_stan = self.data_stan.assign(target = self.y)
            
            if return_:
                return self.data_stan
            else:
                return
        
        material, _ = self.make_array()
        
        self.standard_X = stand_scale.fit_transform(material)
        self.MinMax_X = relu_scale.fit_transform(material)
        
        if return_:
            return self.standard_X, self.MinMax_X


    def manual_scale(self) :
        
        self.Manual_X, self.y = self.make_array()
        
        for i in range(self.Manual_X.shape[0]) :
            for j in range(self.Manual_X.shape[1]) :
                
                temp_ = self.Manual_X[i,j]
                for k in range(len(temp_)) :
                    temp = temp_[k]
                    if temp > 0.49 :
                        self.Manual_X[i,j][k] = 0
                    else:
                        self.Manual_X[i,j][k] = 1
        
        return self.Manual_X


    def get_best_features(self,feturs_to_slct = 5,
                          solver_ = 'sag',
                          C_ = 0.1,
                          standard_X = False,
                          MinMax_X = False ,
                          manual_X_scaled = False,
                          Just_for_corr = False, 
                          silently = True, 
                          return_ = False) :
        
        
        self.logist_model = LogisticRegression( C = C_, solver = solver_ )
        
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
                self.initial_data(self.tar_col_name)
                return
            else:
                self.data = data
                self.initial_data(self.tar_col_name)
        
        if MinMax_X:
            data_x = pd.DataFrame(self.MinMax_X)
            data_y = pd.Series(self.y)
            data = data_x.assign( target = data_y )
            if Just_for_corr:
                self.data = data
                self.initial_data(self.tar_col_name)
                return
            else:
                self.data = data
                self.initial_data(self.tar_col_name)
        
        if manual_X_scaled:
            data_x = pd.DataFrame(self.Manual_X)
            data_y = pd.Series(self.y)
            data = data_x.assign( target = data_y )
            if Just_for_corr:
                self.data = data
                self.initial_data(self.tar_col_name)
                return
            else:
                self.data = data
                self.initial_data(self.tar_col_name)                 
        
        gbf.fit(X, y)
        
        self.best_cols = list(gbf.support_)
        temp_del_invalid_col = []
        for i in range( len( self.best_cols )) :
            
            temp = self.best_cols[i]
            if temp != True:
                temp_del_invalid_col.append( self.data_columns[i] )
        
                
        self.data = self.data.drop(temp_del_invalid_col, axis=1)
        self.initial_data(self.tar_col_name)
        
        if silently == False:
            print("Selected Features:", gbf.support_)
        
        if return_:
            return self.data


    def models(self, save_ = True, to_save_ = 'data/Faze1/output/CSVs',
               test_size_ = 0.2, n_estimators_ = 1000, ) :
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size_)
        
        rf_classifier = RandomForestClassifier( n_estimators = n_estimators_)
        
        rf_classifier.fit(X_train, y_train)
        
        y_pred = rf_classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        if save_ == False:
            print("Random Forest Accuracy:", accuracy)
            return
        
        y_test = pd.DataFrame(y_test)
        to_save = y_test.assign( y_pred = y_pred )
        id_ = NameHelper()
        to_save.to_csv(f"{to_save_}/RandomForestClassifier{id_}.csv")
        
        print("Random Forest Accuracy:", accuracy)

#end#