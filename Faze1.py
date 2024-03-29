#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
from main import PreProcess

test = PreProcess()

test.load_data()

test.initial_data()

test.fix_data() # Once has been used liked this : test.fix_data(save_=True) and we had output data here -->> output/sorted_data.csv

test.stat_jobs(split_= False, save_=True)

#Final of all -->>  The PrePared DataSet is now available here -->> output/dataframeALL.csv

from new import SKlearn

test = SKlearn()

test.load_data('output/dataframeALL.csv')

test.initial_data()

test.cal_corr()

test.get_best_features(MinMax_X = True) #Best Until Now

test.cal_corr()

test.models()


# #

from new import SKlearn

test = SKlearn()

test.load_data_from_PreProcess()

test.initial_data(target_col_name='emotion')

test.manual_scale()

#NOOOOOOOOT WORKINGYRST! $##
#test.cal_corr()

#test.get_best_features(MinMax_X = True) #Best Until Now

#test.cal_corr()

#test.models()