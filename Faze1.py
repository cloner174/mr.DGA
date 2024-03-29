#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__name__), '..', 'main')))

from main import main

test = main.PreProcess()

test.load_data()

test.initial_data()

test.fix_data()

test.stat_jobs(n_ = 2, 
               save_=True, 
               out_where_='Faze1/output/CSVs/Stat_jobs_n2.csv')

