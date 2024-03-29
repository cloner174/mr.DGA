#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
from warnings import warn
import pandas as pd
import numpy as np
from time import sleep
from main import Stats

class PreProcess :
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
        self.extracted = None
    
    
    def load_data(self, input_ = None, index_col_ = None) :
        
        if input_:
            self.input = input_
        else:
            self.input = 'data/Faze1/input/dataset.csv'
        if index_col_ != None:
            self.data = pd.read_csv(self.input, index_col= index_col_ )
        else:
            self.data = pd.read_csv(self.input)
    
    
    def initial_data(self, 
                     emotion_col = None, 
                     emotion_col_num = None, 
                     data_col = None, 
                     need_sort = True ) :
        
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
        
        if emotion_col_num:
            self.emotions = list( (self.data.iloc[:, self.emotion_col_num]).unique() )
        else:
            self.emotions = list( (self.data.loc[:, self.emotion_col_name]).unique() )
    
    
    def name_helper(self):
        
        import uuid
        self.random_id = uuid.uuid4().hex
    
    
    def save_data(self, data_ = None, name_ = None) :
        
        
        if name_:
            name_ = name_
        else:
            self.name_helper()
            name_ = f"data/Faze1/output/CSVs/data{self.random_id}.csv"
        if data_:
            data_.to_csv(name_)
        else:
            self.sorted_data.to_csv(name_, index= False)
    
    
    def fix_data(self, returnArray = False, 
                 ncol_start = None, 
                 ncol_end = None, 
                 return_ = False, 
                 save_ = False,
                 name_ = None):
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
                
                temp_ =  data_[i, j]
                pix_val = np.array(temp_.split(), dtype=int)
                data_[i, j] = pix_val
        
        if returnArray != False:
            
            data = data_
        
        else:
            data = pd.DataFrame(data_)
            data = data.set_axis( self.col_names, axis= 'columns' )
        
        self.sorted_data = data
        
        if save_:
            self.save_data(name_ = name_)
        
        if return_:
            return self.sorted_data            
    
    
    def stat_jobs(self, split_ = True, 
                  n_ = None, 
                  range_ = None, 
                  dim3_ = False, 
                  save_ = False, 
                  out_where_ = None) :
        
        if n_:
            n = n_
        else:
            n = 7
        
        if range_:
            rows_ = range_
        else:
            rows_ = self.n_rows
        
        
        if split_ == False:
            for i in range(rows_) :
                temp = (self.sorted_data).values[i, self.data_col]
                temp_emo = (self.sorted_data).values[i, self.emotion_col_num]
                temp_stat = Stats.stat(temp)
                self.dataframe.append(temp_stat)
                self.y.append(temp_emo)
            
            data_x = pd.DataFrame(self.dataframe, dtype = np.float64)
            data_y = pd.Series(self.y, dtype = int)
            
            if save_:
                
                if out_where_:
                    
                    out_where = f"{out_where_}/dataframeALL.csv"
                else:
                    
                    out_where = r"data/Faze1/output/CSVs/dataframeALL.csv"
                
                data_x = data_x.set_axis(['Mean','Median','Mode','STD','Variance','Quantile1','Quantile2'], axis=1)
                data = data_x.assign( target = data_y )
                data.to_csv(out_where, index=False)
                print(f" The PrePared DataSet is now available here -->> {out_where}")
                return
            else:
                return np.asarray(data_x, dtype = np.float64), np.asarray(data_y)     
        
        
        for i in range(rows_) :
            
            temp = (self.sorted_data).values[i, self.data_col]
            temp_sub = Stats.sublist(temp, n)
            for any_list in temp_sub:
                
                self.y.append(self.sorted_data.iloc[i,self.emotion_col_num])
                self.sublists.append(any_list)
        
        
        if dim3_:
            
            for any_ in self.sublists:
                temp_stat = Stats.stat(any_)
                self.dataframe.append(temp_stat)
            
            data_x = pd.DataFrame(self.dataframe, dtype = np.float64)
            data_y = pd.Series(self.y, dtype = int)
            
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
                return np.asarray(data_x, dtype = np.float64), np.asarray(data_y, dtype = int)
        
        for any_ in self.sublists:
            
            #print(len(self.sublists))
            temp_stat = Stats.stat(any_)
            self.DataFrame['Mean'].append( np.float64(temp_stat[0]) )
            self.DataFrame['Median'].append( np.float64(temp_stat[1]) )
            self.DataFrame['Mode'].append( np.float64(temp_stat[2] ))
            self.DataFrame['STD'].append( np.float64(temp_stat[3] ))
            self.DataFrame['Variance'].append( np.float64(temp_stat[4] ))
            self.DataFrame['Quantile1'].append( np.float64(temp_stat[5] ))
            self.DataFrame['Quantile2'].append( np.float64(temp_stat[6]) )
        
        
        data_x = pd.DataFrame(self.DataFrame, dtype = np.float64)
        data_y = pd.Series(self.y, dtype = int)        
        
        if save_:
            
            if out_where_:
                
                out_where = out_where_
            else:
                self.name_helper()
                out_where = f"data/Faze1/output/CSVs/data{self.random_id}.csv"           
            
            data = data_x.assign( target = data_y )
            data.to_csv(out_where, index=False)
            print(f" The PrePared DataSet is now available here -->> {out_where}")
            
        else:
            
            return np.asarray(data_x, dtype = np.float64), np.asarray(data_y, dtype = int)
    
    
    def extract_all_with_each_col(self = None,
            data_: dict = None,
            n_col = None,
            save_ = False,
            return_ = False,
            self_needed = False) :
        
        warn("\nThe Only Supported Type for data is Python-Dict , use dict() to retype if its a DataFrame \n", category=UserWarning, stacklevel=2)
        sleep(1.15)
        if self == None:
            print("Ready, wait for data_ and n_col")
            sleep(0.2)
            if data_ == None :
                return
            else:
                print('processing')
                pass
        if data_:
            data = pd.DataFrame(data_)
            if n_col:
                n_col = n_col
            else:
                n_col = self.data_col
        else:
            try:
                data = self.sorted_data
                n_col = self.data_col
            except:
                raise ValueError( " There was an Error , try using main.PreProcess.fix_data() func or main.PreProcess.initial_data() first !")
        
        extracted = data.iloc[:, n_col].apply(pd.Series)
        pixel_count = [ i+1 for i in range( extracted.shape[1] ) ]
        new_cols = [ f'pixel_{j}' for j in pixel_count ]
        extracted.columns = new_cols
        temp_df = pd.concat([data, extracted], axis=1)
        data_new = temp_df.drop(data.columns[n_col], axis=1)
        if save_ :
            if self != None:
                self.save_data( data_ = data_new, name_= 'data/Faze1/output/CSVs/AllWithEachCol.csv')
                if self_needed:
                    self.extracted = extracted
                if return_:
                    return extracted
            else:
                data_new.to_csv()
        
        elif return_:
            if self_needed:
                if self != None:
                    self.extracted = extracted
                else:
                    pass
            return extracted
        
        elif self_needed:
            self.extracted = extracted
            return
        else:
            print(" This ProceSS iS EXecUtable !")
            return

#end#
