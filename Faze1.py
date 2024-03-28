#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
from main import PreProcess

test = PreProcess()

#test.load_data()        # we run it once before to get the data below !
                         # The data from defult value was given to fix_data() function !

test.load_data(input_= 'data/sorted_data_pre.csv',index_col_ = 0)

#test.initial_data()     # We run it once before
                         # The dataset is not sorted by defult!

test.initial_data(need_sort=False)

test.fix_data(save_=True)

# test.stat_jobs(dim3_=True, save_=True) #  -->> This is for when need 3dim!
# test.stat_jobs(save_=True)
#x_, y_ = test.run()
#print(x_.shape, y_.shape)