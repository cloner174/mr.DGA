#         #           #              In The Name Of GOD   #
#
#cloner174.org@gmail.com
#github.com/cloner174
#
from main.SKjobs import SKlearn
from main.preprocess import PreProcess
PATHes_from_before = {
    'Stat_jobs_n2' : 'data/Faze1/output/CSVs/Stat_jobs_n2.csv',
    'Stat_jobs_noSplit' : 'data/Faze1/output/CSVs/Stat_jobs_noSplit.csv',
    'AllWithEachCol' : 'data/Faze1/output/CSVs/AllWithEachCol.csv'
}
test = SKlearn()

test.load_data(PATHes_from_before['Stat_jobs_n2'])

test.initial_data()

test.cal_corr()

test.get_best_features()

test.cal_corr()

test.models()