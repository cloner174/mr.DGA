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

test.stat_jobs(split_=False,
               save_=True, 
               out_where_='data/Faze1/output/CSVs/Stat_jobs_noSplit.csv')