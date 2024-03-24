#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
from main import PreProcess
import numpy as np

test = PreProcess()

test.load_data(input_= 'data/sorted_data_pre.csv',index_col_ = 0)

test.initial_data(need_sort=False)

#test.fix_data()

test.json_fix()

test.run(save_=True)

#test_out.to_csv('output/test_out.csv')

#print(test_out)
