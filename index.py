#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
from main import PreProcess


test = PreProcess()

test.load_data(input_= 'sorted_data_pre.csv',index_col_ = 0)

test.initial_data(need_sort=False)

#test.fix_data()

test.json_fix()

test_out = test.run(range_= 1000)

print(test_out)