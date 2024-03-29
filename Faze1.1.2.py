#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#

from main.preprocess import PreProcess

test = PreProcess()

test.load_data()

test.initial_data()

test.fix_data()

test.extract_all_with_each_col(save_=True)

print('done')