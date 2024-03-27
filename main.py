#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
import pandas as pd
import numpy as np
from scipy import stats
import json
import re



class Stats:
    
    def sublist(List, n):
        
        li = List
        len_ = len(li)
        max_index = ( len_ - 1 )
        splitCoef = int( round(  ( len_ / n ), 0 ) )
        split_index = list( ( i for i in range(0, max_index, splitCoef) ) )
        
        ll = []
        for i in range( len(split_index)):
            te = split_index[i]
            if i == 0:
                sub1 = li[:te+1]
                if len(sub1) >= 2:
                  ll.append(li[:te+1])
            else:
               if i == len(split_index)-1:
                   ll.append( li[te:] )
               elif te < max_index:
                   ll.append( ( li[  split_index[i-1] : split_index[i]  ]  ) )
               else:
                   pass
        
        return ll
    
    
    def stat(subset, quantiles_ = None):
        #Q = input( " Please Inter the prob Number for quantiles ...... Leave it blank for defult .. \n   You may inter here .....--->>>>.....")
        #if Q == '':
        #    quantiles = [0.2, 0.25, 0.27, 0.3, 0.33, 0.35, 0.36, 0.60, 0.61, 0.64, 0.66, 0.69, 0.72, 0.75, 0.95]
        #elif Q ==  ' ':
        #    quantiles = [0.2, 0.25, 0.27, 0.3, 0.33, 0.35, 0.36, 0.60, 0.61, 0.64, 0.66, 0.69, 0.72, 0.75, 0.95]
        #else:
        #    try:
        #        quantiles = list(json.loads(Q))
        #    except:
        #        raise ValueError( " Sorry , Something is not right ! ")
        if quantiles_:
            quantiles = quantiles_
        else:
            #quantiles = [0.5, 0.1, 0.2, 0.3, 0.35, 0.40, 0.42, 0.44, 0.48, 0.52, 0.57, 0.65, 0.70, 0.75, 0.85, 0.95]
            quantiles = [0.25, 0.75]

        
        return [
            np.mean(subset),
            np.median(subset),
            stats.mode(subset)[0],
            np.std(subset),
            np.var(subset),
            *[np.quantile(subset, q) for q in quantiles]
        ]



class PreProcess:
    
    def __init__(self) :
        
        self.data = None
        self.n_rows = None
        self.n_cols = None
        self.col_names = None
        self.data_col = None
        self.sorted_data = None
        self.input = None
        self.emotion_col_name = None
        self.emotion_col_num = None
        self.emotions = None
        self.data_col = None
        self.disturbu = []
        self.y = []
        self.sublists = []
        self.dataframe = []
        self.DataFrame = {
            'Mean' : [],
            'Median' : [],
            'Mode' : [],
            'STD' : [],
            'Variance' : [],
            'Quantile1' : [],
            'Quantile2' : []
        }
    
    def load_data(self, input_ = None, index_col_ = None) :
        
        if input_:
            self.input = input_
        else:
            self.input = 'dataset.csv'
        if index_col_ != None:
            self.data = pd.read_csv(self.input, index_col= index_col_ )
        else:
            self.data = pd.read_csv(self.input)
    
    def initial_data(self, emotion_col = None,emotion_col_num = None, data_col = None, need_sort = True ) :
        
        self.n_cols = self.data.shape[1]
        self.n_rows = self.data.shape[0]
        self.col_names = list(self.data.columns)
        
        if emotion_col:
            
            self.emotion_col_name = emotion_col
            self.emotion_col_num = emotion_col_num
        else:
            
            self.emotion_col_name = 'emotion'
            self.emotion_col_num = 0
            
        if data_col:
            
            self.data_col = data_col
        else:
            
            self.data_col = 1
        if need_sort != True :
            
            self.sorted_data = self.data
        else:
            
            self.sorted_data = self.data.sort_values( by = self.emotion_col_name )
        
        self.emotions = list( (self.data.loc[:, self.emotion_col_name]).unique() )
    
    def fix_data(self) :
        
        print(" This may take a few while,, \n  Sit tite and wait till its done! ")
        for i in range(self.n_rows) :
            
            (self.sorted_data).iloc[i, self.data_col] = re.sub('\D',r',', (self.sorted_data).iloc[i, self.data_col])
            (self.sorted_data).iloc[i, self.data_col] = re.sub("^", r"[", (self.sorted_data).iloc[i, self.data_col])
            (self.sorted_data).iloc[i, self.data_col] = re.sub("$", r"]", (self.sorted_data).iloc[i, self.data_col])
        
        print("All Done, \n  Since it has been a little tricky doing this process, All the generated data from this particlur function \n is in output folder ")
        self.sorted_data.to_csv('sorted_data_pre.csv')
    
    def json_fix(self, returArray = False, ncol_start = None, ncol_end = None, return_ = False):
        #                                                            
        #                      # Not Inplace, You should assing it to a variable to store changes #
        data = self.sorted_data
        data_ = np.array(data)
        if ncol_start:
            start_ = ncol_start
        else:
            start_ = 1
        if ncol_end:
            end_ = ncol_end
        else:
            end_ = int( data_.shape[1] ) #Columns
        
        for i in range( 0,  self.n_rows ):  #Rows
            
            for j in range(start_, end_):                    #Columns
                
                temp_ =  json.loads( data_[i, j] )
                data_[i, j] = temp_
        
        if returArray != False:
            
            data = data_
        
        else:
            data = pd.DataFrame(data_)
            data = data.set_axis( self.col_names, axis= 'columns' )
        
        self.sorted_data = data
        
        if return_:
            return self.sorted_data
    
    def run(self, n_ = None, range_ = None, dim3_ = False, save_ = False, out_where_ = None) :
        
        if n_:
            n = n_
        else:
            n = 7
        
        if range_:
            rows_ = range_
        else:
            rows_ = self.n_rows
        
        for i in range(rows_) :
            
            temp = (self.sorted_data).values[i, self.data_col]
            temp_sub = Stats.sublist(temp, n)
            for any_list in temp_sub:
                
                self.y.append(self.sorted_data.iloc[i,self.emotion_col_num])
                self.sublists.append(any_list)

        if dim3_:
            
            for any in self.sublists:
                temp_stat = Stats.stat(any)
                self.dataframe.append(temp_stat)
            
            data_x = pd.DataFrame(self.dataframe, dtype = np.float64)
            data_y = pd.Series(self.y, dtype = np.float64)
            
            if save_:
                
                if out_where_:
                    
                    out_where = out_where_
                else:
                    
                    out_where = r"output/dataframe.csv"

                data = data_x.assign( target = data_y )
                data.to_csv(out_where, index=False)
                print(f" The PrePared DataSet is now available here -->> {out_where}")
                return
            
            else:
                return np.asarray(data_x, dtype = np.float64), np.asarray(data_y, dtype = np.float64)
        
        for any in self.sublists:
            
            #print(len(self.sublists))
            temp_stat = Stats.stat(any)
            self.DataFrame['Mean'].append( np.float64(temp_stat[0]) )
            self.DataFrame['Median'].append( np.float64(temp_stat[1]) )
            self.DataFrame['Mode'].append( np.float64(temp_stat[2] ))
            self.DataFrame['STD'].append( np.float64(temp_stat[3] ))
            self.DataFrame['Variance'].append( np.float64(temp_stat[4] ))
            self.DataFrame['Quantile1'].append( np.float64(temp_stat[5] ))
            self.DataFrame['Quantile2'].append( np.float64(temp_stat[6]) )
        
        
        data_x = pd.DataFrame(self.DataFrame, dtype = np.float64)
        data_y = pd.Series(self.y, dtype = np.float64)        
        
        if save_:
            
            if out_where_:
                
                out_where = out_where_
            else:
                
                out_where = r"output/data.csv"           
            
            data = data_x.assign( target = data_y )
            data.to_csv(out_where, index=False)
            print(f" The PrePared DataSet is now available here -->> {out_where}")
            
        else:
            
            return np.asarray(data_x, dtype = np.float64), np.asarray(data_y, dtype = np.float64)