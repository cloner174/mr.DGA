#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from main.stats import NameHelper
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SKlearn :
    
    
    def __init__(self) :
        
        self.data = None
        self.data_Standard = None
        self.data_MinMax = None
        self.data_Manual = None
        self.X = None
        self.y = None
        self.corr_ = None
        self.best_cols = None
        self.data_columns = None
        self.Standard_X = None
        self.MinMax_X = None
        self.Manual_X = None
    
    
    def load_data(self, data_path) :
        
        self.data = data_path
        self.data = pd.read_csv(self.data)
    
    
    def load_data_from_PreProcess(self, 
                                    call_extracted_all = False) :
        
        from main import PreProcess
        
        pre_process = PreProcess()
        pre_process.load_data()
        pre_process.initial_data()
        if call_extracted_all:
            pre_process.fix_data()
            self.data = pre_process.extract_all_with_each_col(return_=True)
        else:
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
                    standard_x = False,
                    minmax_x = False,
                    manual_x = False,
                    return_ = False,
                    title_ = 'Correlation Matrix',
                    figsize_ = (10, 8),
                    save_corr = False,
                    save_corr_where = 'data/Faze1/output/CSVs/cal_corr',
                    save_fig = False,
                    save_fig_where = 'data/Faze1/output/Pics/cal_corr',
                    name_ = None) :
        
        if standard_x:
            self.corr_ = self.data_Standard.corr()
        elif minmax_x:
            self.corr_ = self.data_MinMax.corr()
        elif manual_x:
            self.corr_ = self.data_Manual.corr()
        else:
            self.corr_ = self.data.corr()
        
        if show_ == False:
            if return_:
                return self.corr_
            else:
                return
        
        plt.figure( figsize = figsize_)
        sns.heatmap(self.corr_, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
        plt.title(title_)
        
        if save_fig:
            if name_:
                id_ = name_
            else:
                id_ = NameHelper()
            if save_corr:
                self.corr_.to_csv(f"/{save_corr_where}/{id_}.csv")
            plt.savefig(f"{save_fig_where}/{id_}.jpg")
        
        plt.show()
    
    
    def scaling(self,
                return_ = False,
                standard_x = False,#                 Will Not Work if return_ = False
                minmax_x = False,#                   Will Not Work if return_ = False
                also_data_scalled = False,#          Will Not Work if return_ = False
                all_ = False,#                       Will Not Work if return_ = False
                need_y = False) :#                   Just Works with standard_x or minmax_x
        
        stand_scale = StandardScaler()
        relu_scale =  MinMaxScaler()
        
        material, y_ = self.make_array()
        
        Standard_X = stand_scale.fit_transform(material)
        MinMax_X = relu_scale.fit_transform(material)
        
        temp = pd.DataFrame(self.Standard_X)
        data_Standard = temp.assign(target = self.y)
        
        temp = pd.DataFrame(self.MinMax_X)
        data_MinMax = temp.assign(target = self.y)
        
        if return_:
            if standard_x:
                if also_data_scalled:
                    return Standard_X, data_Standard
                else:
                    if need_y:
                        return Standard_X, y_
                    else:
                        return Standard_X
            elif minmax_x:
                if also_data_scalled:            
                    return MinMax_X, data_MinMax
                else:
                    if need_y:
                        return MinMax_X, y_
                    else:
                        return MinMax_X
            elif all_:
                if also_data_scalled:
                    return Standard_X, MinMax_X, data_Standard, data_MinMax
                else:
                    return Standard_X, MinMax_X
            else:
                return
        else:
            self.Standard_X = Standard_X
            self.MinMax_X = MinMax_X
            self.data_Standard = data_Standard
            self.data_MinMax = data_MinMax
    
    
    def manual_scale(self,
                        need_y = False) :
        
        Manual_X, y_ = self.make_array()
        
        for i in range(Manual_X.shape[0]) :
            for j in range(Manual_X.shape[1]) :
                
                temp_ = Manual_X[i,j]
                for k in range(len(temp_)) :
                    temp = temp_[k]
                    if temp > 0.49 :
                        Manual_X[i,j][k] = 0
                    else:
                        Manual_X[i,j][k] = 1
        
        self.Manual_X = Manual_X
        if need_y:
            return y_
        else:
            return self.Manual_X
    
    
    def get_best_features(self,feturs_to_slct = 5,
                            solver_ = 'sag',
                            C_ = 0.1,
                            standard_x = False,
                            minmax_x = False ,
                            manual_x = False,
                            Just_for_corr = False, 
                            silently = True, 
                            return_ = False) :
        
        warnings.simplefilter("ignore", category=FutureWarning)
        
        logist_model = LogisticRegression( C = C_, solver = solver_ )
        
        gbf = RFE( estimator = self.logist_model, 
                    n_features_to_select = feturs_to_slct)
        
        if standard_x:
            X, y = self.scaling(return_ = True, standard_x = True, need_y = True)
        elif minmax_x:
            X, y = self.scaling(return_ = True, minmax_x = True, need_y = True)
        elif manual_x:
            y = self.manual_scale(need_y = True)#  It will save X in self.Manual_X and return the dummy y
            X = self.Manual_X
        else:
            X, y = self.make_array()
        #################
        data_x = pd.DataFrame(self.Standard_X)
        data_y = pd.Series(self.y)
        data = data_x.assign( target = data_y )
        if standard_x:
            data_x = pd.DataFrame(self.Standard_X)
            data_y = pd.Series(self.y)
            data = data_x.assign( target = data_y )
            if Just_for_corr:
                self.data = data
                self.initial_data(self.tar_col_name)
                return
            else:
                self.data = data
                self.initial_data(self.tar_col_name)
        
        if minmax_x:
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
        
        if manual_x:
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
